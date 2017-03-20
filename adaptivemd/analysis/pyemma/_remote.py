from adaptivemd import Model


def remote_analysis(
        files,
        topfile='input.pdb',
        tica_lag=2,
        tica_dim=2,
        msm_states=5,
        msm_lag=2,
        stride=1):
    """
    Remote analysis function to be called by the RPC Python call

    Parameters
    ----------
    files : list of `Trajectory`
        a list of `Trajectory` objects
    topfile : `File`
        a reference to the `.pdb` file using in pyemma
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
    import pyemma
    import numpy as np

    feat = pyemma.coordinates.featurizer(topfile)

    selection = feat.select_Heavy()
    feat.add_selection(selection)

    pyemma.config.show_progress_bars = False

    # todo: allow specification of several folders and wildcats, used for session handling
    # if isinstance(trajfiles, basestring):
    #     if '*' in trajfiles or trajfiles.endswith('/'):
    #         files = glob.glob(trajfiles)

    print '#files :', len(files)

    inp = pyemma.coordinates.source(files, feat)

    #tica_obj = pyemma.coordinates.tica(
    #    inp, lag=tica_lag, dim=tica_dim, kinetic_map=False)

    y = inp.get_output()

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
        cl = pyemma.coordinates.cluster_regspace(data, dmin=dmin, stride=stride, metric='minRMSD')
        for cc in range(n_try_max):
            n_cl_now = cl.n_clusters
            delta_cl_now = np.abs(n_cl_now - n_clusters_target)
            if not n_clusters_target-err <= cl.n_clusters <= n_clusters_target+err:
                # Cheap (and sometimes bad) heuristic to get relatively close relatively quick
                dmin = cl.dmin*cl.n_clusters/   n_clusters_target
                cl = pyemma.coordinates.cluster_regspace(data, dmin=dmin, metric='minRMSD', max_centers=5000, stride=stride)# max_centers is given so that we never reach it (dangerous)
            else:
                break
            if verbose:
                print('cl iter %u %u -> %u (Delta to target (%u +- %u): %u'%(cc, n_cl_now, cl.n_clusters,
                                                                             n_clusters_target, err, delta_cl_now))
        return cl
    ### END GUILLE'S CODE

    cl = regspace_cluster_to_target(y, msm_states,
                                    delta=int(msm_states/10),
                                    stride=stride)
    m = pyemma.msm.estimate_markov_model(cl.dtrajs, msm_lag)

    #cl = pyemma.coordinates.cluster_kmeans(data=y, k=msm_states, stride=stride)


    data = {
        'input': {
            'frames': inp.n_frames_total(),
            'dimension': inp.dimension(),
            'n_trajectories': inp.number_of_trajectories(),
            'lengths': inp.trajectory_lengths(),
        },

        'clustering': {
            'k': msm_states,
            'dtrajs': [
                t for t in cl.dtrajs
            ]
        },
        'msm': {
            'lagtime': msm_lag,
            'P': m.P,
            'C': m.count_matrix_full
        }
    }

    return Model(data)
