# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
from collections import OrderedDict
from typing import Dict, Tuple

import config

import torch
from nvflare.apis.fl_component import FLComponent

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants, DefaultCheckpointFileName, EnvironmentKey
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.model_desc import ModelDescriptor
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager


class PTFileModelPersistorGAN(FLComponent):
    def __init__(
            self,
            exclude_vars=None,
            model_gen=None,
            model_disc=None,
            source_ckpt_file_full_name_gen=None,
            source_ckpt_file_full_name_disc=None,
            filter_id: str = None,
            global_model_file_name_gen=DefaultCheckpointFileName.GLOBAL_MODEL,
            global_model_file_name_disc=DefaultCheckpointFileName.GLOBAL_MODEL,
            best_global_model_file_name_gen=DefaultCheckpointFileName.GLOBAL_MODEL,
            best_global_model_file_name_disc=DefaultCheckpointFileName.GLOBAL_MODEL,
    ):
        """Persist pytorch-based model to/from file system.

        This Model Persistor tries to load PT model data in the following three ways:

            1. Load from a specified source checkpoint file
            2. Load from a location from the app folder
            3. Load from a torch model object

        The Persistor tries method 1 first if the source_ckpt_file_full_name is specified;
        If source_ckpt_file_full_name is not specified, it tries method 2;
        If no checkpoint location is specified in the app folder, it tries method 3.

        Method 2 - Load from a location from the app folder

        It is assumed that the app folder must contain the environments.json file. Among other things, this
        JSON file must specify where to find the checkpoint file. It does so with two JSON elements:

            - APP_CKPT_DIR: specifies the folder (within the app) where the checkpoint file resides.
            - APP_CKPT: specifies the base file name of the checkpoint

        Here is an example of the environments.json content::

            {
                "APP_CKPT_DIR": "model",
                "APP_CKPT": "pretrained_model.pt"
            }

        In this example, the checkpoint file is located in the "model" folder within the app and is named
        pretrained_model.pt.

        Method 3 - Load from a torch model object. In this case, the 'model' arg must be a valid torch
        model, or the component ID of a valid torch model included in the "components" section of
        your config_fed_server.json.

        If all 3 methods fail, system_panic() is called.

        If checkpoint folder name is specified, then global model and best global model will be saved to it;
        Otherwise they will be saved directly in the app folder.

        The model is saved in a dict depending on the persistor you used. You might need to access it with
        ``model.load_state_dict(torch.load(path_to_model)["model"])`` as there is additional meta information together with the model weights.

        Args:
            exclude_vars (str, optional): regex expression specifying weight vars to be excluded from training. Defaults to None.
            model (str, optional): torch model object or component id of the model object. Defaults to None.
            global_model_file_name (str, optional): file name for saving global model. Defaults to DefaultCheckpointFileName.GLOBAL_MODEL.
            best_global_model_file_name (str, optional): file name for saving best global model. Defaults to DefaultCheckpointFileName.BEST_GLOBAL_MODEL.
            source_ckpt_file_full_name (str, optional): full file name for source model checkpoint file. Defaults to None.
            filter_id: Optional string that defines a filter component that is applied to prepare the model to be saved,
                e.g. for serialization of custom Python objects.
        Raises:
            ValueError: when source_ckpt_file_full_name does not exist
        """
        super().__init__(
        )
        print("INITIALIZING_PERSISTOR")
        self.exclude_vars = re.compile(exclude_vars) if exclude_vars else None
        self.model_gen = model_gen
        self.model_disc = model_disc
        self.log_dir = None
        self.ckpt_preload_path = None
        self.persistence_manager_gen = None
        self.persistence_manager_disc = None
        self.ckpt_dir_env_key = EnvironmentKey.CHECKPOINT_DIR
        self.ckpt_file_name_env_key = EnvironmentKey.CHECKPOINT_FILE_NAME
        # self.global_model_file_name = global_model_file_name
        # self.best_global_model_file_name = best_global_model_file_name
        # self.source_ckpt_file_full_name_gen = source_ckpt_file_full_name_gen
        # self.source_ckpt_file_full_name_disc = source_ckpt_file_full_name_disc
        self.source_ckpt_file_full_name_gen = global_model_file_name_gen
        self.source_ckpt_file_full_name_disc = global_model_file_name_disc

        self.global_model_file_name_gen = global_model_file_name_gen
        self.global_model_file_name_disc = global_model_file_name_disc
        self.best_global_model_file_name_gen = best_global_model_file_name_gen
        self.best_global_model_file_name_disc = best_global_model_file_name_disc

        self.default_train_conf_gen = None
        self.default_train_conf_disc = None

        if source_ckpt_file_full_name_gen and not os.path.exists(source_ckpt_file_full_name_gen):
            raise ValueError("specified source checkpoint model file {} does not exist")
        if source_ckpt_file_full_name_disc and not os.path.exists(source_ckpt_file_full_name_disc):
            raise ValueError("specified source checkpoint model file {} does not exist")

    def _initialize(self, fl_ctx: FLContext):

        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        env = None
        run_args = fl_ctx.get_prop(FLContextKey.ARGS)
        # if run_args:
        #     env_config_file_name = os.path.join(app_root, run_args.env)
        #     if os.path.exists(env_config_file_name):
        #         try:
        #             with open(env_config_file_name) as file:
        #                 env = json.load(file)
        #         except BaseException:
        #             self.system_panic(
        #                 reason="error opening env config file {}".format(env_config_file_name), fl_ctx=fl_ctx
        #             )
        #             return

        # self.log_info(fl_ctx, "ENV: " + str(env))
        # if env is not None:
        #     if env.get(self.ckpt_dir_env_key, None):
        #         fl_ctx.set_prop(AppConstants.LOG_DIR, env[self.ckpt_dir_env_key], private=True, sticky=True)
        #     if env.get(self.ckpt_file_name_env_key) is not None:
        #         fl_ctx.set_prop(
        #             AppConstants.CKPT_PRELOAD_PATH, env[self.ckpt_file_name_env_key], private=True, sticky=True
        #         )

        log_dir = fl_ctx.get_prop(AppConstants.LOG_DIR)
        if log_dir:
            self.log_dir = os.path.join(app_root, log_dir)
        else:
            # self.log_info(fl_ctx, "APP_ROOT: " + str(app_root))
            self.log_dir = app_root

        # self.log_info(fl_ctx, "LOG_DIR: " + str(self.log_dir))
        # self.log_info(fl_ctx, "GLOBAL MODEL FILENAME: " + str(self.global_model_file_name_gen))

        # self._ckpt_save_path_gen = os.path.join(self.log_dir, self.global_model_file_name_gen)
        # self._ckpt_save_path_disc = os.path.join(self.log_dir, self.global_model_file_name_disc)
        self._ckpt_save_path_gen = os.path.join(config.MODELS_DIR, self.global_model_file_name_gen)
        self._ckpt_save_path_disc = os.path.join(config.MODELS_DIR, self.global_model_file_name_disc)

        self._best_ckpt_save_path_gen = os.path.join(self.log_dir, self.best_global_model_file_name_gen)
        self._best_ckpt_save_path_disc = os.path.join(self.log_dir, self.best_global_model_file_name_disc)

        self.source_ckpt_file_full_name_gen = self._ckpt_save_path_gen
        self.source_ckpt_file_full_name_disc = self._ckpt_save_path_disc

        self.log_info(fl_ctx, "CKPT_SAVE_PATH_GEN: " + str(self._ckpt_save_path_gen))
        # self.log_info(fl_ctx, "BEST_SAVE_PATH: " + str(self._best_ckpt_save_path))
        ckpt_preload_path = fl_ctx.get_prop(AppConstants.CKPT_PRELOAD_PATH)
        if ckpt_preload_path:
            self.ckpt_preload_path = os.path.join(app_root, ckpt_preload_path)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(config.MODELS_DIR):
            os.makedirs(config.MODELS_DIR)

        if isinstance(self.model_gen, str):
            # treat it as model component ID
            model_component_id = self.model_gen
            engine = fl_ctx.get_engine()
            self.model_gen = engine.get_component(model_component_id)
            if not self.model_gen:
                self.system_panic(reason="cannot find model component '{}'".format(model_component_id), fl_ctx=fl_ctx)
                return
            if not isinstance(self.model_gen, torch.nn.Module):
                self.system_panic(
                    reason="expect model component '{}' to be torch.nn.Module but got {}".format(
                        model_component_id, type(self.model_gen)
                    ),
                    fl_ctx=fl_ctx,
                )
                return
        elif self.model_gen and not isinstance(self.model_gen, torch.nn.Module):
            self.system_panic(
                reason="expect model to be torch.nn.Module but got {}".format(type(self.model_gen)), fl_ctx=fl_ctx
            )
            return

        if isinstance(self.model_disc, str):
            # treat it as model component ID
            model_component_id = self.model_disc
            engine = fl_ctx.get_engine()
            self.model_disc = engine.get_component(model_component_id)
            if not self.model_disc:
                self.system_panic(reason="cannot find model component '{}'".format(model_component_id), fl_ctx=fl_ctx)
                return
            if not isinstance(self.model_disc, torch.nn.Module):
                self.system_panic(
                    reason="expect model component '{}' to be torch.nn.Module but got {}".format(
                        model_component_id, type(self.model_disc)
                    ),
                    fl_ctx=fl_ctx,
                )
                return
        elif self.model_disc and not isinstance(self.model_disc, torch.nn.Module):
            self.system_panic(
                reason="expect model to be torch.nn.Module but got {}".format(type(self.model_disc)), fl_ctx=fl_ctx
            )
            return

        fl_ctx.sync_sticky()

    def load_models(self, fl_ctx: FLContext) -> Tuple[ModelLearnable, ModelLearnable]:
        """Convert initialised model into Learnable/Model format.

        Args:
            fl_ctx (FLContext): FL Context delivered by workflow

        Returns:
            Model: a Learnable/Model object
        """
        src_file_name_gen = None
        src_file_name_disc = None

        if config.LOAD_MODEL and self.source_ckpt_file_full_name_disc and self.source_ckpt_file_full_name_gen:
            src_file_name_gen = self.source_ckpt_file_full_name_gen
            src_file_name_disc = self.source_ckpt_file_full_name_disc

        if src_file_name_gen and src_file_name_disc:
            try:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                data_gen = torch.load(src_file_name_gen, map_location=device)
                data_disc = torch.load(src_file_name_disc, map_location=device)

                self.log_info(fl_ctx, "Using checkpoints")
                # "checkpoint may contain 'model', 'optimizer', 'lr_scheduler', etc. or only contain model dict directly."
            except BaseException:
                self.log_exception(fl_ctx, "error loading checkpoint from {} or {}".format(src_file_name_gen,
                                                                                           src_file_name_disc))
                self.system_panic(reason="cannot load model checkpoint", fl_ctx=fl_ctx)
                return None
        else:
            # if no pretrained model provided, use the generated network weights from APP config
            # note that, if set "determinism" in the config, the init model weights will always be the same
            try:
                data_gen = self.model_gen.state_dict() if self.model_gen is not None else OrderedDict()
                data_disc = self.model_disc.state_dict() if self.model_disc is not None else OrderedDict()

                self.log_info(fl_ctx, "Initialized models")
            except BaseException:
                self.log_exception(fl_ctx, "error getting state_dict from model object")
                self.system_panic(reason="cannot create state_dict from model object", fl_ctx=fl_ctx)
                return None

        if self.model_gen:
            self.default_train_conf_gen = {"train": {"model": type(self.model_gen).__name__}}
        if self.model_disc:
            self.default_train_conf_disc = {"train": {"model": type(self.model_disc).__name__}}

        self.persistence_manager_gen = PTModelPersistenceFormatManager(data_gen,
                                                                       default_train_conf=self.default_train_conf_gen)
        self.persistence_manager_disc = PTModelPersistenceFormatManager(data_disc,
                                                                        default_train_conf=self.default_train_conf_disc)
        gen_ml = self.persistence_manager_gen.to_model_learnable(self.exclude_vars)
        disc_ml = self.persistence_manager_disc.to_model_learnable(self.exclude_vars)

        return gen_ml, disc_ml

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)
        elif event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE:
            # save the current model as the best model!
            self.save_model_file_gen(self._best_ckpt_save_path_gen)
            self.save_model_file_disc(self._best_ckpt_save_path_disc)

    def save_model_file_gen(self, save_path: str):
        save_dict = self.persistence_manager_gen.to_persistence_dict()
        torch.save(save_dict, save_path)

    def save_model_file_disc(self, save_path: str):
        save_dict = self.persistence_manager_disc.to_persistence_dict()
        torch.save(save_dict, save_path)

    def save_models(self, gen_ml: ModelLearnable, disc_ml: ModelLearnable, fl_ctx: FLContext):
        if config.SAVE_MODEL:
            self.log_info(fl_ctx, 'Saving models to ' + self._ckpt_save_path_gen)
            self.persistence_manager_gen.update(gen_ml)
            self.persistence_manager_disc.update(disc_ml)
            self.save_model_file_gen(self._ckpt_save_path_gen)
            self.save_model_file_disc(self._ckpt_save_path_disc)

    def get_model(self, model_file: str, fl_ctx: FLContext, model: str) -> ModelLearnable:
        try:
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Use the "cpu" to load the global model weights, avoid GPU out of memory
            device = "cpu"
            location = os.path.join(self.log_dir, model_file)
            data = torch.load(location, map_location=device)
            if model == 'gen':
                persistence_manager = PTModelPersistenceFormatManager(data,
                                                                      default_train_conf=self.default_train_conf_gen)
            else:
                persistence_manager = PTModelPersistenceFormatManager(data,
                                                                      default_train_conf=self.default_train_conf_disc)

            return persistence_manager.to_model_learnable(self.exclude_vars)
        except BaseException:
            self.log_exception(fl_ctx, "error loading checkpoint from {}".format(model_file))
            return {}

    def get_model_inventory(self, fl_ctx: FLContext) -> Dict[str, ModelDescriptor]:
        model_inventory = {}
        location = os.path.join(self.log_dir, self.global_model_file_name_gen)
        if os.path.exists(location):
            model_inventory[self.global_model_file_name_gen] = ModelDescriptor(
                name=self.global_model_file_name_gen,
                location=location,
                model_format=self.persistence_manager_gen.get_persist_model_format(),
                props={},
            )

        location = os.path.join(self.log_dir, self.best_global_model_file_name_gen)
        if os.path.exists(location):
            model_inventory[self.best_global_model_file_name_gen] = ModelDescriptor(
                name=self.best_global_model_file_name_gen,
                location=location,
                model_format=self.persistence_manager_gen.get_persist_model_format(),
                props={},
            )

        location = os.path.join(self.log_dir, self.global_model_file_name_disc)
        if os.path.exists(location):
            model_inventory[self.global_model_file_name_disc] = ModelDescriptor(
                name=self.global_model_file_name_disc,
                location=location,
                model_format=self.persistence_manager_disc.get_persist_model_format(),
                props={},
            )

        location = os.path.join(self.log_dir, self.best_global_model_file_name_disc)
        if os.path.exists(location):
            model_inventory[self.best_global_model_file_name_disc] = ModelDescriptor(
                name=self.best_global_model_file_name_disc,
                location=location,
                model_format=self.persistence_manager_disc.get_persist_model_format(),
                props={},
            )

        return model_inventory
