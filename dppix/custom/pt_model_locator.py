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

import os
from typing import List, Union

import torch.cuda
from pt_constants import PTConstants
from generator_model import Generator
from discriminator_model import Discriminator

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import model_learnable_to_dxo
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager


class PTModelLocator(ModelLocator):
    def __init__(self):
        super().__init__()
        self.gen = Generator()
        self.disc = Discriminator()

    def get_model_names(self, fl_ctx: FLContext) -> List[str]:
        return [PTConstants.PTServerName]

    def locate_model(self, model_name, fl_ctx: FLContext) -> Union[DXO, None]:
        if model_name == PTConstants.PTServerName:
            try:
                server_run_dir = fl_ctx.get_engine().get_workspace().get_app_dir(fl_ctx.get_job_id())
                gen_model_path = os.path.join(server_run_dir, PTConstants.PTFileGenModelName)
                disc_model_path = os.path.join(server_run_dir, PTConstants.PTFileDiscModelName)

                if not os.path.exists(gen_model_path) or not os.path.exists(disc_model_path):
                    return None

                # Load the torch model
                device = "cuda" if torch.cuda.is_available() else "cpu"
                data_gen = torch.load(gen_model_path, map_location=device)

                # Set up the persistence manager.
                if self.gen:
                    default_train_conf = {"train": {"model": type(self.gen).__name__}}
                else:
                    default_train_conf = None

                # Use persistence manager to get learnable
                persistence_manager = PTModelPersistenceFormatManager(data_gen, default_train_conf=default_train_conf)
                ml_gen = persistence_manager.to_model_learnable(exclude_vars=None)

                # DISC
                data_disc = torch.load(disc_model_path, map_location=device)

                # Set up the persistence manager.
                if self.disc:
                    default_train_conf = {"train": {"model": type(self.disc).__name__}}
                else:
                    default_train_conf = None

                # Use persistence manager to get learnable
                persistence_manager = PTModelPersistenceFormatManager(data_disc, default_train_conf=default_train_conf)
                ml_disc = persistence_manager.to_model_learnable(exclude_vars=None)

                # Create dxo and return
                gen_dxo = model_learnable_to_dxo(ml_gen)
                disc_dxo = model_learnable_to_dxo(ml_disc)

                return DXO(data_kind=DataKind.COLLECTION, data={'gen': gen_dxo, 'disc': disc_dxo})

            except Exception as e:
                self.log_error(fl_ctx, f"Error in retrieving {model_name}: {e}.", fire_event=False)
                return None
        else:
            self.log_error(fl_ctx, f"PTModelLocator doesn't recognize name: {model_name}", fire_event=False)
            return None
