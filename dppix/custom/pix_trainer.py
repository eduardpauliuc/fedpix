import os.path

import torch
from azureml.core import Workspace, Dataset

from nvflare.apis.event_type import EventType
from tqdm import tqdm

import config
from dataset import MapDataset
from utils import save_some_examples, save_results
from pt_constants import PTConstants
from generator_model import Generator
from discriminator_model import Discriminator
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode, FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager

# from azureml.core import Workspace, Dataset

class PixTrainer(Executor):
    def __init__(
            self,
            data_path="~/data",
            lr=0.0002,
            epochs=5,
            train_task_name=AppConstants.TASK_TRAIN,
            submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
            exclude_vars=None,
            pre_train_task_name=AppConstants.TASK_GET_WEIGHTS,
            dataset_name="maps-dataset",
    ):
        """Cifar10 Trainer handles train and submit_model tasks. During train_task, it trains a
        simple network on CIFAR10 dataset. For submit_model task, it sends the locally trained model
        (if present) to the server.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            train_task_name (str, optional): Task name for train task. Defaults to "train".
            submit_model_task_name (str, optional): Task name for submit model. Defaults to "submit_model".
            exclude_vars (list): List of variables to exclude during model loading.
            pre_train_task_name: Task name for pre train task, i.e., sending initial model weights.
        """
        super().__init__()

        self._lr = lr
        self._epochs = epochs
        self._train_task_name = train_task_name
        self._pre_train_task_name = pre_train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars

        # Training setup
        # self.model = SimpleNetwork()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        # self.loss = nn.CrossEntropyLoss()
        # self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9)

        self.disc = Discriminator(in_channels=3).to(config.DEVICE)
        self.gen = Generator(in_channels=3, features=64).to(config.DEVICE)
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999), )
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
        self.BCE = nn.BCEWithLogitsLoss()
        self.L1_LOSS = nn.L1Loss()

        # Create Cifar10 dataset for training.
        # transforms = Compose(
        #     [
        #         ToTensor(),
        #         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #     ]
        # )
        # self._train_dataset = CIFAR10(root=data_path, transform=transforms, download=True, train=True)
        # self._train_loader = DataLoader(self._train_dataset, batch_size=4, shuffle=True)

        self._train_dataset = None
        self._train_loader = None

        self._g_scaler = torch.cuda.amp.GradScaler()
        self._d_scaler = torch.cuda.amp.GradScaler()
        self._val_dataset = None
        self._val_loader = None
        self._n_iterations = None

        self._evaluation_folder = None
        self._results_folder = None

        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager
        # in case no initial model is found.
        self._gen_default_train_conf = {"train": {"model": type(self.gen).__name__}}
        self.gen_persistence_manager = PTModelPersistenceFormatManager(
            data=self.gen.state_dict(), default_train_conf=self._gen_default_train_conf
        )
        self._disc_default_train_conf = {"train": {"model": type(self.disc).__name__}}
        self.disc_persistence_manager = PTModelPersistenceFormatManager(
            data=self.disc.state_dict(), default_train_conf=self._disc_default_train_conf
        )

        self.workspace = None
        self.dataset_name = dataset_name

    def setup(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, 'PREPARING')
        client_name = fl_ctx.get_identity_name()

        self.workspace = Workspace.from_config()

        maps_dataset = Dataset.get_by_name(
            self.workspace, self.dataset_name)
        maps_dataset.download(target_path=config.TRAIN_DIR, overwrite=True)

        self._train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
        self._train_loader = DataLoader(
            self._train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
        )

        val_dir = os.path.join(config.VAL_DIR, client_name)
        self._val_dataset = MapDataset(root_dir=val_dir)
        self._val_loader = DataLoader(self._val_dataset, batch_size=1, shuffle=True)
        self._n_iterations = len(self._train_loader)

        self._evaluation_folder = os.path.join(config.EVALUATION_DIR, client_name)
        if not os.path.exists(self._evaluation_folder):
            os.makedirs(self._evaluation_folder)

        self._results_folder = os.path.join(config.RESULTS_DIR, client_name)
        if not os.path.exists(self._results_folder):
            os.makedirs(self._results_folder)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.setup(fl_ctx)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            if task_name == self._pre_train_task_name:
                # Get the new state dict and send as weights
                return self._get_model_weights()
            elif task_name == self._train_task_name:
                # Get model weights
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data kind is weights.
                if not dxo.data_kind == DataKind.COLLECTION:
                    self.log_error(fl_ctx, f"data_kind expected COLLECTION but got {dxo.data_kind} instead.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                print("In TRAIN task, the DXO received is", dxo)

                # Convert weights to tensor. Run training
                gen_weights = {k: torch.as_tensor(v) for k, v in dxo.data['gen'].data.items()}
                disc_weights = {k: torch.as_tensor(v) for k, v in dxo.data['disc'].data.items()}
                self._local_train(fl_ctx, gen_weights, disc_weights, abort_signal)

                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Save the local model after training.
                self._save_local_model(fl_ctx)

                # Get the new state dict and send as weights
                return self._get_model_weights()
            elif task_name == self._submit_model_task_name:
                # Load local model
                gen, disc = self._load_local_models(fl_ctx)

                # Get the model parameters and create dxo from it
                gen_dxo = model_learnable_to_dxo(gen)
                disc_dxo = model_learnable_to_dxo(disc)
                return DXO(data_kind=DataKind.COLLECTION, data={'gen': gen_dxo, 'disc': disc_dxo}).to_shareable()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in simple trainer: {e}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _get_model_weights(self) -> Shareable:
        # Get the new state dict and send as weights
        gen_weights = {k: v.cpu().numpy() for k, v in self.gen.state_dict().items()}
        disc_weights = {k: v.cpu().numpy() for k, v in self.disc.state_dict().items()}

        gen_dxo = DXO(
            data_kind=DataKind.WEIGHTS, data=gen_weights, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations}
        )

        disc_dxo = DXO(
            data_kind=DataKind.WEIGHTS, data=disc_weights, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations}
        )

        outgoing_dxo = DXO(
            data_kind=DataKind.COLLECTION, data={'gen': gen_dxo, 'disc': disc_dxo},
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations}
        )

        return outgoing_dxo.to_shareable()

    def _local_train(self, fl_ctx, gen_weights, disc_weights, abort_signal):
        # Set the model weights
        self.gen.load_state_dict(state_dict=gen_weights)
        self.disc.load_state_dict(state_dict=disc_weights)

        # shared_context = fl_ctx.get_prop(FLContextKey.PEER_CONTEXT)

        # Basic training
        self.gen.train()
        self.disc.train()

        results_queue = []

        for epoch in range(self._epochs):
            results = self.train_fn(fl_ctx, epoch, abort_signal)

            results_queue.append(results)

            # run_number = shared_context.get_prop(AppConstants.CURRENT_ROUND)

            if config.SAVE_MODEL and epoch % 5 == 0 or epoch == self._epochs - 1:
                save_results(results_queue, folder=self._results_folder)
                results_queue = []

            save_some_examples(self.gen, self._val_loader, epoch, folder=self._evaluation_folder)

            # running_loss = 0.0
            # for i, batch in enumerate(self._train_loader):
            #     if abort_signal.triggered:
            # If abort_signal is triggered, we simply return.
            # The outside function will check it again and decide steps to take.
            # return

            # images, labels = batch[0].to(self.device), batch[1].to(self.device)
            # self.optimizer.zero_grad()

            # predictions = self.model(images)
            # cost = self.loss(predictions, labels)
            # cost.backward()
            # self.optimizer.step()

            # running_loss += cost.cpu().detach().numpy() / images.size()[0]
            # if i % 3000 == 0:
            #     self.log_info(
            #         fl_ctx, f"Epoch: {epoch}/{self._epochs}, Iteration: {i}, " f"Loss: {running_loss / 3000}"
            #     )
            #     running_loss = 0.0

            # MINE
            # if config.SAVE_MODEL and epoch % 5 == 0:
            #     print("Saved")
            #     save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            #     save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
            #

    def train_fn(self, fl_ctx, epoch, abort_signal):
        # loop = tqdm(self._train_loader, leave=True)

        d_loss = 0
        dr_loss = 0
        df_loss = 0
        gl1_loss = 0
        gf_loss = 0
        g_loss = 0

        for idx, (x, y) in enumerate(self._train_loader):
            if abort_signal.triggered:
                # If abort_signal is triggered, we simply return.
                # The outside function will check it again and decide steps to take.
                return

            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            # Train Discriminator
            with torch.cuda.amp.autocast():
                y_fake = self.gen(x)
                D_real = self.disc(x, y)
                D_real_loss = self.BCE(D_real, torch.ones_like(D_real))
                D_fake = self.disc(x, y_fake.detach())
                D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

            self.opt_disc.zero_grad()
            self._d_scaler.scale(D_loss).backward()
            self._d_scaler.step(self.opt_disc)
            self._d_scaler.update()

            # Train generator
            with torch.cuda.amp.autocast():
                D_fake = self.disc(x, y_fake)
                G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake))
                L1 = self.L1_LOSS(y_fake, y) * config.L1_LAMBDA
                G_loss = G_fake_loss + L1

            self.opt_gen.zero_grad()
            self._g_scaler.scale(G_loss).backward()
            self._g_scaler.step(self.opt_gen)
            self._g_scaler.update()

            if idx % 5 == 0:
                self.log_info(
                    fl_ctx, f"Epoch: {epoch}/{self._epochs}, Iteration: {idx}, "
                            f"G_loss: {G_loss.cpu().item()}"
                )

                # loop.set_postfix(
                #     D_real=torch.sigmoid(D_real).mean().item(),
                #     D_fake=torch.sigmoid(D_fake).mean().item(),
                #     D_real_loss=D_real_loss.item(),
                #     D_fake_loss=D_fake_loss.item()
                # )

            dr_loss += D_real_loss.cpu().item()
            df_loss += D_fake_loss.cpu().item()
            d_loss += D_loss.cpu().item()
            gl1_loss += L1.cpu().item()
            gf_loss += G_fake_loss.cpu().item()
            g_loss += G_loss.cpu().item()

        return (d_loss / len(self._train_loader), dr_loss / len(self._train_loader),
                df_loss / len(self._train_loader), g_loss / len(self._train_loader),
                gf_loss / len(self._train_loader), gl1_loss / len(self._train_loader))

    def _save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        gen_path = os.path.join(models_dir, PTConstants.PTLocalGenModelName)
        disc_path = os.path.join(models_dir, PTConstants.PTLocalDiscModelName)

        gen_ml = make_model_learnable(self.gen.state_dict(), {})
        disc_ml = make_model_learnable(self.disc.state_dict(), {})
        self.gen_persistence_manager.update(gen_ml)
        self.disc_persistence_manager.update(disc_ml)
        torch.save(self.gen_persistence_manager.to_persistence_dict(), gen_path)
        torch.save(self.disc_persistence_manager.to_persistence_dict(), disc_path)

    def _load_local_models(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None

        gen_path = os.path.join(models_dir, PTConstants.PTLocalGenModelName)
        disc_path = os.path.join(models_dir, PTConstants.PTLocalDiscModelName)

        self.gen_persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(gen_path), default_train_conf=self._gen_default_train_conf
        )
        self.disc_persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(disc_path), default_train_conf=self._disc_default_train_conf
        )

        gen = self.gen_persistence_manager.to_model_learnable(exclude_vars=self._exclude_vars)
        disc = self.disc_persistence_manager.to_model_learnable(exclude_vars=self._exclude_vars)
        return gen, disc
