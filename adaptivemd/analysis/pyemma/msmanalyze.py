import pyemma.coordinates as coor
import pyemma.msm as msm

import numpy as np
import argparse
from sys import exit
from pyemma import config

import json

import logging

logging.disable(logging.CRITICAL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze a number of files and compute an MSM')

    parser.add_argument(
        'file',
        metavar='input.dcd',
        help='the output .dcd file',
        type=str, nargs='+')

    parser.add_argument(
        '-c', '--tica-lagtime', dest='tica_lag',
        type=int, default=2, nargs='?',
        help='the lagtime used for tica')

    parser.add_argument(
        '-d', '--tica-dimensions', dest='tica_dim',
        type=int, default=2, nargs='?',
        help='the lagtime used for tica')

    parser.add_argument(
        '-s', '--stride', dest='stride',
        type=int, default=1, nargs='?',
        help='the lagtime used for tica')

    parser.add_argument(
        '-l', '--msm-lagtime', dest='msm_lag',
        type=int, default=2, nargs='?',
        help='the lagtime used for the final msm')

    parser.add_argument(
        '-k', '--msm-states', dest='msm_states',
        type=int, default=5, nargs='?',
        help='number of k means centers and number of msm states')

    parser.add_argument(
        '-t', '--topology', dest='topology_pdb',
        type=str, default='topology.pdb', nargs='?',
        help='the path to the topology.pdb file')

    parser.add_argument(
        '-v', '--verbose',
        dest='verbose', action='store_true',
        default=False,
        help='if set then text output is send to the ' +
             'console.')

    args = parser.parse_args()

    # Load files / replace by linked files

    trajfiles = args.file
    topfile = args.topology_pdb

    print trajfiles

    # Choose parameters to be used in the task

    config.show_progress_bars = False

    lag = args.tica_lag

    feat = coor.featurizer(topfile)
    selection = feat.select_Heavy()
    feat.add_selection(selection)

    # print feat.describe()

    inp = coor.source(trajfiles, feat)
    # print 'trajectory length = ', inp.trajectory_length(0)
    # print 'number of dimension = ', inp.dimension()

    dim = args.tica_dim

    tica_obj = coor.tica(inp, lag=lag, dim=dim, kinetic_map=False)
    Y = tica_obj.get_output()

    # print 'Mean values: ', np.mean(Y, axis=0)
    # print 'Variances:   ', np.var(Y, axis=0)

    # print -lag / np.log(tica_obj.eigenvalues[:dim])



    ### BEGIN GUILLE'S CODE
    def regspace_cluster_to_target(data, n_clusters_target,
                                   n_try_max=5, delta=5.,
                                   verbose=False, stride=1):
        r"""
        Clusters a dataset to a target n_clusters using regspace clustering by iteratively. "
        Work best with 1D data
        data: ndarray or list thereof
        n_clusters_target: int, number of clusters.
        n_try_max: int, default is 5. Maximum number of iterations in the heuristic.
        delta: float, defalut is 5. Percentage of n_clusters_target to consider converged.
                 Eg. n_clusters_target=100 and delta = 5 will consider any clustering between 95 and 100 clustercenters as
                 valid. Note. Note: An off-by-one in n_target_clusters is sometimes unavoidable
        returns: pyemma clustering object
        tested:True
        """
        delta = delta/100

        assert np.vstack(data).shape[0] >= n_clusters_target, "Cannot cluster " \
                                                          "%u datapoints on %u clustercenters. Reduce the number of target " \
                                                          "clustercenters."%(np.vstack(data).shape[0], n_clusters_target)
        # Works well for connected, 1D-clustering,
        # otherwise it's bad starting guess for dmin
        # cmax = np.vstack(data).max()
        # cmin = np.vstack(data).min()
        # dmin = (cmax-cmin)/(n_clusters_target+1)
        dmin = .2
        err = np.ceil(n_clusters_target*delta)
        cl = coor.cluster_regspace(data, dmin=dmin, stride=stride, metric='minRMSD')
        for cc in range(n_try_max):
            n_cl_now = cl.n_clusters
            delta_cl_now = np.abs(n_cl_now - n_clusters_target)
            if not n_clusters_target-err <= cl.n_clusters <= n_clusters_target+err:
                # Cheap (and sometimes bad) heuristic to get relatively close relatively quick
                dmin = cl.dmin*cl.n_clusters/   n_clusters_target
                cl = coor.cluster_regspace(data, dmin=dmin, metric='minRMSD', max_centers=5000, stride=stride)# max_centers is given so that we never reach it (dangerous)
            else:
                break
            if verbose:
                print('cl iter %u %u -> %u (Delta to target (%u +- %u): %u'%(cc, n_cl_now, cl.n_clusters,
                                                                             n_clusters_target, err, delta_cl_now))
        return cl
    ### END GUILLE'S CODE

    # clr = coor.cluster_regspace(data=Y, dmin=0.5)
    # cl = coor.cluster_kmeans(data=Y, k=args.msm_states, stride=args.stride)
    cl = regspace_cluster_to_target(Y, args.msm_states,
                                    delta=int(args.msm_states/10),
                                    stride=args.stride)

    M = msm.estimate_markov_model(cl.dtrajs, args.msm_lag)

    # print 'fraction of states used = ', M.active_state_fraction
    # print 'fraction of counts used = ', M.active_count_fraction

    # print M.timescales()

    # print cl.dtrajs

    # os.makedirs('dtrajs/')

    with open("model.dtraj", "w") as f:
        f.write("\n".join(" ".join(map(str, x)) for x in cl.dtrajs))

    # np.savetxt("model.dtraj", cl.dtrajs, delimiter=" ", fmt='%d')
    np.savetxt("model.msm", M.P, delimiter=",")

    # print M.P

    data = {
        'input': {
            'frames': inp.n_frames_total(),
            'dimension': inp.dimension(),
            'trajectories': inp.number_of_trajectories(),
            'lengths': inp.trajectory_lengths().tolist(),
        },
        'tica': {
            'dimension': tica_obj.dimension()
        },
        'clustering': {
            'dtrajs': [
                t.tolist() for t in cl.dtrajs
            ]
        },
        'msm': {
            'P': M.P.tolist()
        }
    }

    print json.dumps(data)

    exit(0)
