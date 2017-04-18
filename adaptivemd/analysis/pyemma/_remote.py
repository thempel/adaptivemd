##############################################################################
# adaptiveMD: A Python Framework to Run Adaptive Molecular Dynamics (MD)
#             Simulations on HPC Resources
# Copyright 2017 FU Berlin and the Authors
#
# Authors: Jan-Hendrik Prinz
# Contributors:
#
# `adaptiveMD` is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MDTraj. If not, see <http://www.gnu.org/licenses/>.
##############################################################################


# The remote function to be called py PyEMMAAnalysis


def remote_analysis(
        trajectory_paths,
        selection=None,
        features=None,
        topfile='input.pdb',
        trajectory_objects=None,
        tica_lag=2,
        tica_dim=2,
        msm_states=5,
        msm_lag=2,
        stride=1):
    """
    Remote analysis function to be called by the RPC Python call

    Parameters
    ----------
    trajectory_paths : Trajectory file paths
    selection : str
        an atom subset selection string as used in mdtraj .select
    features : dict or list or None
        a feature descriptor in the format. A dict has exactly one entry:
        functionname: [attr1, attr2, ...]. attributes can be results of
        function calls. All function calls are to the featurizer object!
        If a list is given each element is considered to be a feature
        descriptor. If None (default) all coordinates will be added as
        features (.add_all())

        Examples

            {'add_backbone_torsions': None}
            -> feat.add_backbone_torsions()

            {'add_distances': [ [[0,10], [2,20]] ]}
            -> feat.add_distances([[0,10], [2,20]])

            {'add_inverse_distances': [
                { 'select_backbone': None } ]}
            -> feat.add_inverse_distances(select_backbone())

    topfile : `File`
        a reference to the full topology `.pdb` file using in pyemma
    tica_lag : int
        the lagtime used for tCIA
    tica_dim : int
        number of dimensions using in tICA. This refers to the number of tIC used
    msm_states : int
        number of microstates used for the MSM
    msm_lag : int
        lagtime used for the MSM construction
    stride : int
        a stride to be used on the data. Can speed up computation at reduced accuracy

    Returns
    -------
    `Model`
        a model object with a data attribute which is a dict and contains all relevant
        information about the computed MSM
    """
    import os

    import pyemma
    import mdtraj as md
    import itertools
    import numpy as np
    pdb = md.load(topfile)
    topology = pdb.topology

    if selection:
        topology = topology.subset(topology.select(selection_string=selection))

    feat = pyemma.coordinates.featurizer(topology)

    def simplex_surface(traj, atom_indices=[589, 2086, 2087, 2088]):
        assert len(atom_indices) == 4

        _inp = md.compute_distances(traj, [[a, b] for a, b in itertools.combinations(atom_indices, 2)])

        distances2triangles = [[0, 3, 1],
                               [1, 5, 2],
                               [3, 4, 5],
                               [0, 4, 2]]

        simplex_surfs = np.array([np.sqrt(_inp[:, t].sum(axis=1) / 2 * \
                                          (_inp[:, t].sum(axis=1) / 2 - _inp[:, t][:, 0]) * \
                                          (_inp[:, t].sum(axis=1) / 2 - _inp[:, t][:, 1]) * \
                                          (_inp[:, t].sum(axis=1) / 2 - _inp[:, t][:, 2])) for t in
                                  distances2triangles]).sum(axis=0)

        return simplex_surfs.reshape(simplex_surfs.shape[0], 1)

    feat.add_custom_func(simplex_surface, 1)
    
    inp = pyemma.coordinates.source(trajectory_paths, feat)
    y = inp.get_output()


    cl = pyemma.coordinates.cluster_regspace(data=y, dmin=.1, stride=stride)
    m = pyemma.msm.estimate_markov_model(cl.dtrajs, msm_lag)

    data = {
        'input': {
            'n_atoms': topology.n_atoms,
            'frames': inp.n_frames_total(),
            'n_trajectories': inp.number_of_trajectories(),
            'lengths': inp.trajectory_lengths(),
            'selection': selection
        },
        'features': {
            'features': features,
            'feat.describe': feat.describe(),
            'n_features': inp.dimension(),
        },
        'clustering': {
            'n_clusters': cl.n_clusters,
            'dtrajs': [
                t for t in cl.dtrajs
            ],
            'clustercenters': cl.clustercenters
        },
        'msm': {
            'lagtime': msm_lag,
            'P': m.P,
            'C': m.count_matrix_full
        }
    }

    return data
