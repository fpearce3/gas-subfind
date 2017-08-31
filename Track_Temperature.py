import os
import median_properties as mp
import catalogue as cat
import numpy as np
import halo as hl
import profiles as pr
import create_maps as cm
#import track_ids as ti
from mpi4py import MPI


def find_haloes(path):
    halos = []
    for x in os.listdir(path):
        if x.startswith('halo_'): halos.append('{0}/{1}'.format(path, x))
    return sorted(halos)


if __name__ == "__main__":

    Bahamas = False
    Eagle   = False
    EagleNC = True

    maps = False

    comm   = MPI.COMM_WORLD
    NProcs = comm.Get_size()
    Rank   = comm.Get_rank()

    if Bahamas:
        path = '/cosma5/data/dp004/dc-barn1/work/CE_LowRes/gas_runs'
        outfile = 'bahamas'
        snap = '029'
    elif Eagle:
        path = '../../../data/eagle'
        outfile = 'eagle'
        snap = '029_z000p000'
    elif EagleNC:
        path = '../../../data/eagle_nc'
        outfile = 'eagle_nc'
        snap = '029_z000p000'
    else:
        if not Rank: print ' > Exiting'
        quit()

#    snap = '029'
    if not Rank: print ' > Halo directory: {0}/'.format(path)

    halos = find_haloes(path)
    if Rank == 0: print ' > Found {0:d} haloes'.format(len(halos))

    med_props = mp.median_properties(halos)

    catalogue = cat.create_catalogue('{0}/tmp_{1:04d}.hdf5'.format(outfile, Rank))

    for j in xrange(Rank, len(halos), NProcs):
        if not Rank: print '--- {0}'.format(halos[j].split('/')[-1])

#        h = hl.halo(halos[j], outfile='eagle/Properties_Sn029.hdf5'.format(Rank))
#        h = hl.halo(halos[j], outfile='eagle_nc/Properties_Sn029.hdf5'.format(Rank))
#        h = hl.halo(halos[j], outfile='bahamas/Properties_Sn029.hdf5'.format(Rank))
        h = hl.halo(halos[j], snapshot=snap)
        if h.Success is False:
            print '{0}: READ ERROR {1}'.format(MyRank, halos[j])
            continue
        if not Rank: print ' > Data read'

        """ Initialize various classes """
        Maps = cm.maps(h, FoV=10.0, Nx=512, Ny=512, Depth=10.0)
        Tprofiles = pr.profiles(h)

        """ Compute quantities of interest from here """
        if not Rank: print ' > Cumulative mass profile'
        Tprofiles.cum_mass(h)

        if not Rank: print ' > Mass weighted profile'
        Tprofiles.mass_weighted_profile(h)

        if not Rank: print ' > Spectroscopic-like profile'
        Tprofiles.spectroscopic_like_profile(h)

        if not Rank: print ' > Spectroscopic profile'
        my_spectrum = Tprofiles.spectroscopic_Xray_profile(h)

        if not Rank: print ' > Hydrostatic mass estimate'
        Tprofiles.hydrostatic_mass_analysis(h)

#        if Rank == 0: print ' > Profiles plot'
#        Tprofiles.compare_Dprofiles(h)
#        Tprofiles.compare_Tprofiles(h)

        if not Rank: print ' > Radial mass fraction'
        Tprofiles.radial_mass_fraction(h)

        if not Rank: print ' > Radial emission measure fraction'
        Tprofiles.radial_EMM_fraction(h)

        if not Rank: print ' > Saving halo properties'
        catalogue.save_aperture_quantities(h, my_spectrum, Tprofiles, Amin=0.0, \
                                       Amax=h.R500, grp='True_Props', \
                                       tag='_500')
        catalogue.save_aperture_quantities(h, my_spectrum, Tprofiles, Amin=0.15*h.R500, \
                                       Amax=h.R500, grp='True_Props', \
                                       tag='_500ce')
        catalogue.save_aperture_quantities(h, my_spectrum, Tprofiles, Amin=0.0, \
                                       Amax=Tprofiles.R500_hse, \
                                       grp='HSE_Props', tag='_500')
        catalogue.save_aperture_quantities(h, my_spectrum, Tprofiles, \
                                       Amin=0.15*Tprofiles.R500_hse, \
                                       Amax=Tprofiles.R500_hse, \
                                       grp='HSE_Props', tag='_500ce')
        catalogue.save_aperture_quantities(h, my_spectrum, Tprofiles, Amin=0.0, \
                                       Amax=Tprofiles.R500_spec, \
                                       grp='SPEC_Props', tag='_500')
        catalogue.save_aperture_quantities(h, my_spectrum, Tprofiles, \
                                       Amin=0.15*Tprofiles.R500_spec,\
                                       Amax=Tprofiles.R500_spec, \
                                       grp='SPEC_Props', tag='_500ce')
        catalogue.save_profiles(h, my_spectrum, Tprofiles)

        if maps:
            if not Rank: print ' > Mass maps'
            Maps.generic_map(h, h.mass/1.989e33)

        if not Rank: 
            print ' > Storing profiles, mass fraction, emission measure & mass'
        med_props.store(j, h, Tprofiles)

        del h, Tprofiles
        
#    if Rank == 0: print '--- Median properties'
    med_props.sum_cores(comm, NProcs, Rank)

    if not Rank:
        print '--- Finishing'
        print ' > Merging saved outputs'
        catalogue.merge_saves('{0}/halo_properties_{1}.hdf5'.format(outfile, snap))

#    if Rank == 0: med_props.calculate_binned_medians()

    comm.Barrier()
