# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PTB LM small size config.
"""

# pylint: disable=invalid-name, too-few-public-methods, missing-docstring

init_scale = 0.1
num_epochs = 13
hidden_size = 400
keep_prob = 1.0
batch_size = 20
num_steps = 10

cell = {
    "type": "LSTMBlockCell",
    "kwargs": {
        "num_units": hidden_size,
        "forget_bias": 0.
    },
    "dropout": {"output_keep_prob": keep_prob},
    "num_layers": 2
}
emb = {
    "dim": hidden_size
}
opt = {
    "optimizer": {
        "type": "MomentumOptimizer",
        "kwargs": {
            "learning_rate": 0.01,
            "momentum": 0.9
        }
    }
}
