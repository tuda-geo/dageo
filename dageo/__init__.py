# Copyright 2024 D. Werthmüller, G. Serrao Seabra, F.C. Vossepoel
#
# This file is part of dageo.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.

from dageo import utils
from dageo.data_assimilation import esmda, esmda_subspace
from dageo.geertsma import Geertsma, GeertsmaFullGrid
from dageo.particle_filter import particle_filter
from dageo.reservoir_simulator import RandomPermeability, Simulator
from dageo.utils import Report, localization_matrix

# Initialize a random number generator.
rng = utils.rng()


__all__ = [
    "reservoir_simulator",
    "data_assimilation",
    "utils",
    "geertsma",
    "particle_filter",
    "esmda",
    "esmda_subspace",
    "particle_filter",
    "Simulator",
    "RandomPermeability",
    "Geertsma",
    "GeertsmaFullGrid",
    "TorchGeertsma",
    "TorchGeertsmaFullGrid",
    "localization_matrix",
    "rng",
    "Report",
]

__version__ = utils.__version__
