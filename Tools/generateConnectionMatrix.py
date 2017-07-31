'''
This scripts provide a set functions to build connection matrices which are not possible based on boolean logic.
Function: connect(population1, population2, kernel, overlap, camera, save_path, check, save, forceFlag)
        Input: The initiliase neuron populations
'''

from brian2 import *
import matplotlib.pyplot as plt
import px2ind as p
import Connectivity_visualizer as cv 
import os.path


def lowPass(population1, population2, kernel, overlap, camera='DVS128', save_path='/media/moritz/Data/Connections/', check=False):
    save_path = save_path + camera[:-3] + '/'
    filename = save_path + 'LowPass_%s_%s_%s.npy' % (str(kernel[0]), str(kernel[1]), str(overlap))
    # Check if connection file already exists and wether we want to force to recreate it if it already exists
    # if os.path.isfile(filename) and forceFlag:
    #   print 'Loading existing connection matrix: %s_%s_%s_%s_%s.npy' %(population1.name,population2.name, str(kernel[0]), str(kernel[1]), str(overlap))
    #   connections = np.load(filename).astype(int)
    #   if check:
    #       S = Synapses(population1, population2)
    #       S.connect(connections[:,0], connections[:,1])
    #       print 'Visualisation'
    #       cv.visualise_connectivity(S)
    #   return connections

    # # initiliase basic equation in order to check if connection was done right
    # tau = 10*ms
    # eqs = '''
    # dv/dt = (2-v)/tau : 1
    # '''
    S = Synapses(population1, population2)
    connections_source = []
    connections_target = []
    resolution = np.int(camera[-3:])
    target = 0
    hw = np.int(np.floor(kernel[0] / 2.0))
    resolution_y = population1.N / resolution

    if kernel[0] == kernel[1]:
        for source in range(resolution * hw + overlap, population1.N - resolution * (hw) - overlap, kernel[0] - overlap):
            for offset_y in range(-resolution * np.int(np.floor(kernel[1] / 2.0)), resolution * np.int(np.floor(kernel[1] / 2.0)) + 1, resolution):
                for offset_x in range(-np.int(np.floor(kernel[0] / 2.0)), np.int(np.ceil(kernel[0] / 2.0))):
                    if check:
                        S.connect(source + offset_x + offset_y, target)

                    connections_source.append(source + offset_x + offset_y)
                    connections_target.append(target)
            target = target + 1

    elif kernel[0] < kernel[1]:
        # for source in range(resolution*np.int(np.floor(kernel[1]/2.0))+overlap, population1.N-resolution*(np.int(np.floor(kernel[1]/2.0)))-overlap, kernel[0]-overlap):
        for layers in range(0, population1.N / resolution, kernel[1]):
            for source in range(resolution * np.int(np.floor(kernel[1] / 2.0)) + kernel[0] + overlap, resolution * np.int(np.floor(kernel[1] / 2.0)) + resolution - kernel[0] - overlap, kernel[0] - overlap):
                for offset_y in range(-resolution * np.int(np.floor(kernel[1] / 2.0)), resolution * np.int(np.floor(kernel[1] / 2.0)) - 1, resolution):
                    for offset_x in range(-np.int(np.floor(kernel[0] / 2.0)), np.int(np.ceil(kernel[0] / 2.0))):
                        if check:
                            S.connect(source + offset_x + offset_y + layers * resolution, target)

                        connections_source.append(source + offset_x + offset_y + layers * resolution)
                        connections_target.append(target)
                        # print 'Source: %s | Target: %s' %(str(source+offset_x+offset_y), str(target))
                target = target + 1

    elif kernel[0] > kernel[1]:
        # for layers in range(0, population1.N/resolution, kernel[1]):
        # for layers in range(0, population1.N/resolution_y, kernel[0]):
            # for source in range(0+overlap, resolution_y-overlap, kernel[1]-overlap):
        for source in range(resolution * (kernel[1] / 2) + np.int(np.floor(kernel[0] / 2.0)), population1.N - (resolution * (kernel[1] / 2) + np.int(np.floor(kernel[0] / 2.0))) + 1, kernel[0] * kernel[1] - overlap * resolution):
            for offset_y in range(-np.int(np.floor(kernel[1] / 2.0)), np.int(np.ceil(kernel[1] / 2.0))):
                for offset_x in range(-np.int(np.floor(kernel[0] / 2.0)), np.int(np.ceil(kernel[0] / 2.0))):
                    if check:
                        S.connect(source + offset_x + offset_y * resolution, target)

                    connections_source.append(source + offset_x + offset_y * np.int(np.floor(kernel[0] / 2.0)))
                    connections_target.append(target)
                    # print 'Source: %s | Target: %s' %(str(source+offset_x+offset_y), str(target))
            target = target + 1

    if check:
        print ('Visualisation')
        cv.visualise_connectivity(S)

    connections = np.zeros([len(connections_source), 2])
    connections[:, 0] = np.asarray(connections_source)
    connections[:, 1] = np.asarray(connections_target)

    # print 'Saving connection matrix: LowPass_%s_%s_%s.npy' % (str(kernel[0]), str(kernel[1]), str(overlap))
    np.save(filename, connections.astype(int))

    return connections.astype(int)


def orientation(population1, population2, kernel, overlap, camera, save_path):
    offset_x = 0
    if type(camera) == str:
        resolution = np.int(camera[-3:])
    else:
        resolution = camera
    if kernel[0] == 5:
        offset_x = 0
    elif kernel[0] == 7:
        offset_x = 1
    elif kernel[0] == 9:
        offset_x = 2

    # O deg orientated gabor filter
    connections_source_ex = []
    connections_target_ex = []
    connections_source_inh = []
    connections_target_inh = []

    start_row = np.floor(kernel[0] / 2.0).astype(int) * resolution
    col_offset = np.floor(kernel[0] / 2.0).astype(int)
    offset_y = resolution * np.int(np.floor(kernel[1] / 2.0))
    target = 0

    for row_ind in range(start_row, population1.N - resolution - start_row, (kernel[0] - overlap) * resolution):
        for center_ind in range(row_ind + col_offset, row_ind + resolution - 1 - col_offset, kernel[0] - overlap):
            for y in range(-offset_y, offset_y + 1, resolution):
                for x in range(-offset_x, offset_x + 1):
                    source = center_ind + x + y
                    connections_source_ex.append(source)
                    connections_target_ex.append(target)
                    # connect inhibtory synapse
                    if x == -offset_x:
                        connections_source_inh.append(source - 1)
                        connections_target_inh.append(target)
                        connections_source_inh.append(source - 2)
                        connections_target_inh.append(target)
                    if x == offset_x:
                        connections_source_inh.append(source + 1)
                        connections_target_inh.append(target)
                        connections_source_inh.append(source + 2)
                        connections_target_inh.append(target)

            target += 1

    connection_90_on = np.zeros([len(connections_source_ex), 2])
    connection_90_off = np.zeros([len(connections_source_inh), 2])
    connection_90_on[:, 0] = np.asarray(connections_source_ex)
    connection_90_on[:, 1] = np.asarray(connections_target_ex)
    connection_90_off[:, 0] = np.asarray(connections_source_inh)
    connection_90_off[:, 1] = np.asarray(connections_target_inh)

    np.save(save_path + 'gabor_on_%s_%s_90' % (str(kernel[0]), str(kernel[1])), connection_90_on)
    np.save(save_path + 'gabor_off_%s_%s_90' % (str(kernel[0]), str(kernel[1])), connection_90_off)

    # 45 deg orientated gabor filter
    connections_source_ex = []
    connections_target_ex = []
    connections_source_inh = []
    connections_target_inh = []
    target = 0
    for row_ind in range(start_row, population1.N - resolution - start_row, (kernel[0] - overlap) * resolution):
        for center_ind in range(row_ind + col_offset, row_ind + resolution - 1 - col_offset, kernel[0] - overlap):
            # for offset in range(-col_offset, col_offset + 1):
            shift = 0
            for y in range(-offset_y, offset_y + 1, resolution):
                for x in range(-offset_x, offset_x + 1):
                    # source = center_ind + offset + offset * resolution
                    source = center_ind + x + y + shift
                    connections_source_ex.append(source)
                    connections_target_ex.append(target)

                    # if (source - 1 < center_ind - col_offset - col_offset * resolution or
                    #    source + 1 > center_ind + col_offset + col_offset * resolution):
                    #     continue
                    if x == -offset_x:
                        connections_source_inh.append(source - 1)
                        connections_target_inh.append(target)
                        connections_source_inh.append(source - 2)
                        connections_target_inh.append(target)
                    if x == offset_x:
                        connections_source_inh.append(source + 1)
                        connections_target_inh.append(target)
                        connections_source_inh.append(source + 2)
                        connections_target_inh.append(target)
                shift += 1
            target += 1

    connection_45_on = np.zeros([len(connections_source_ex), 2])
    connection_45_off = np.zeros([len(connections_source_inh), 2])
    connection_45_on[:, 0] = np.asarray(connections_source_ex)
    connection_45_on[:, 1] = np.asarray(connections_target_ex)
    connection_45_off[:, 0] = np.asarray(connections_source_inh)
    connection_45_off[:, 1] = np.asarray(connections_target_inh)

    np.save(save_path + 'gabor_on_%s_%s_45' % (str(kernel[0]), str(kernel[1])), connection_45_on)
    np.save(save_path + 'gabor_off_%s_%s_45' % (str(kernel[0]), str(kernel[1])), connection_45_off)

    # 135 deg

    connections_source_ex = []
    connections_target_ex = []
    connections_source_inh = []
    connections_target_inh = []
    target = 0
    for row_ind in range(start_row, population1.N - resolution - start_row, (kernel[0] - overlap) * resolution):
        for center_ind in range(row_ind + col_offset, row_ind + resolution - 1 - col_offset, kernel[0] - overlap):
            shift = kernel[0]
            for y in range(-offset_y, offset_y + 1, resolution):
                for x in range(-offset_x, offset_x + 1):
                    # source = center_ind + offset + offset * resolution
                    source = center_ind + x + y + shift
                    connections_source_ex.append(source)
                    connections_target_ex.append(target)

                    # if (source - 1 < center_ind - col_offset - col_offset * resolution or
                    #    source + 1 > center_ind + col_offset + col_offset * resolution):
                    #     continue
                    if x == -offset_x:
                        connections_source_inh.append(source - 1)
                        connections_target_inh.append(target)
                        connections_source_inh.append(source - 2)
                        connections_target_inh.append(target)
                    if x == offset_x:
                        connections_source_inh.append(source + 1)
                        connections_target_inh.append(target)
                        connections_source_inh.append(source + 2)
                        connections_target_inh.append(target)
                shift -= 1
            # for offset in range(-col_offset, col_offset + 1):
            #     source = center_ind + offset - offset * resolution
            #     connections_source_ex.append(source)
            #     connections_target_ex.append(target)

            #     if (source - 1 == center_ind - col_offset + col_offset * resolution - 1 or
            #        source + 1 == center_ind + col_offset - col_offset * resolution + 1):
            #         continue
            #     connections_source_inh.append(source - 1)
            #     connections_target_inh.append(target)
            #     connections_source_inh.append(source + 1)
            #     connections_target_inh.append(target)
            target += 1

    connection_135_on = np.zeros([len(connections_source_ex), 2])
    connection_135_off = np.zeros([len(connections_source_inh), 2])
    connection_135_on[:, 0] = np.asarray(connections_source_ex)
    connection_135_on[:, 1] = np.asarray(connections_target_ex)
    connection_135_off[:, 0] = np.asarray(connections_source_inh)
    connection_135_off[:, 1] = np.asarray(connections_target_inh)

    np.save(save_path + 'gabor_on_%s_%s_135' % (str(kernel[0]), str(kernel[1])), connection_135_on)
    np.save(save_path + 'gabor_off_%s_%s_135' % (str(kernel[0]), str(kernel[1])), connection_135_off)

    # 0
    connections_source_ex = []
    connections_target_ex = []
    connections_source_inh = []
    connections_target_inh = []
    target = 0
    # col_offset = np.floor(kernel[0] / 2.0).astype(int)
    # offset_y = resolution * np.int(np.floor(kernel[1] / 2.0))

    # offset_x = 1

    for row_ind in range(start_row, population1.N - resolution - start_row, (kernel[0] - overlap) * resolution):
        for center_ind in range(row_ind + col_offset, row_ind + resolution - 1 - col_offset, kernel[0] - overlap):
            for x in range(-col_offset, col_offset + 1):
                for y in range(-offset_x * resolution, offset_x * resolution + 1, resolution):
                    source = center_ind + x + y
                    connections_source_ex.append(source)
                    connections_target_ex.append(target)
                    if y == -offset_x * resolution:
                        connections_source_inh.append(source - resolution)
                        connections_target_inh.append(target)
                        connections_source_inh.append(source - 2 * resolution)
                        connections_target_inh.append(target)
                    if y == offset_x * resolution:
                        connections_source_inh.append(source + resolution)
                        connections_target_inh.append(target)
                        connections_source_inh.append(source + 2 * resolution)
                        connections_target_inh.append(target)
                    # connections_source_inh.append(source - resolution)
                    # connections_target_inh.append(target)
                    # connections_source_inh.append(source + resolution)
                    # connections_target_inh.append(target)

            target += 1

    connection_0_on = np.zeros([len(connections_source_ex), 2])
    connection_0_off = np.zeros([len(connections_source_inh), 2])
    connection_0_on[:, 0] = np.asarray(connections_source_ex)
    connection_0_on[:, 1] = np.asarray(connections_target_ex)
    connection_0_off[:, 0] = np.asarray(connections_source_inh)
    connection_0_off[:, 1] = np.asarray(connections_target_inh)

    np.save(save_path + 'gabor_on_%s_%s_0' % (str(kernel[0]), str(kernel[1])), connection_0_on)
    np.save(save_path + 'gabor_off_%s_%s_0' % (str(kernel[0]), str(kernel[1])), connection_0_off)


def semd(population1, population2, camera='DVS128', save_path='/media/moritz/Data/Connections/'):
    save_path = save_path + camera[:-3] + '/'
    if int(camera[-3:]) == 240:
        aspectRatio = int(camera[-3:]) / 180.0
    else:
        aspectRatio = 1
    resolution = int(np.floor(population1.N / (np.sqrt((population1.N / aspectRatio)))))
    # make horizontal connections
    # normal synapse + nmda synapse (normal synapse adjacent pre synaptic neuron)
    # hw = np.int(np.floor(kernel[0] / 2.0))
    # for source in range(resolution * hw + overlap, population1.N - resolution * (hw) - overlap, kernel[0] - overlap):
    connections_source_l2r = []
    connections_target_l2r = []
    connections_source_r2l = []
    connections_target_r2l = []
    for row_ind in range(0, population1.N - resolution, resolution):
        for ind in range(0, resolution, 2):
            connections_source_l2r.append(ind + row_ind + 1)
            connections_target_l2r.append(ind + row_ind)
            connections_source_r2l.append(ind + row_ind)
            connections_target_r2l.append(ind + row_ind + 1)

    connections_h_l2r = np.zeros([len(connections_source_l2r), 2])
    connections_h_l2r[:, 0] = np.asarray(connections_source_l2r)
    connections_h_l2r[:, 1] = np.asarray(connections_target_l2r)

    connections_h_r2l = np.zeros([len(connections_source_r2l), 2])
    connections_h_r2l[:, 0] = np.asarray(connections_source_r2l)
    connections_h_r2l[:, 1] = np.asarray(connections_target_r2l)

    # Vertical connected semds
    connections_source_l2r = []
    connections_target_l2r = []
    connections_source_r2l = []
    connections_target_r2l = []
    for row_ind in range(0, population1.N - 2 * resolution, 2 * resolution):
        for ind in range(0, resolution):
            connections_source_l2r.append(ind + row_ind + resolution)
            connections_target_l2r.append(ind + row_ind)
            connections_source_r2l.append(ind + row_ind)
            connections_target_r2l.append(ind + row_ind + resolution)

    connections_v_l2r = np.zeros([len(connections_source_l2r), 2])
    connections_v_l2r[:, 0] = np.asarray(connections_source_l2r)
    connections_v_l2r[:, 1] = np.asarray(connections_target_l2r)

    connections_v_r2l = np.zeros([len(connections_source_r2l), 2])
    connections_v_r2l[:, 0] = np.asarray(connections_source_r2l)
    connections_v_r2l[:, 1] = np.asarray(connections_target_r2l)

    #  Saving
    np.save(save_path + 'nmda_h_l2r', connections_h_l2r.astype(int))
    np.save(save_path + 'nmda_h_r2l', connections_h_r2l.astype(int))
    np.save(save_path + 'nmda_v_l2r', connections_v_l2r.astype(int))
    np.save(save_path + 'nmda_v_r2l', connections_v_r2l.astype(int))

    return connections_h_l2r.astype(int), connections_h_r2l.astype(int), connections_v_l2r.astype(int), connections_v_r2l.astype(int)

def rcn(population1, population2, probability, save_path):
    for i in range(3):
        connections_source = []
        connections_target = []
        p = np.random.rand(population1.N * population2.N)
        #
        for source in range(0, population1.N):
            print ('Source: %i/%i' % (source, population1.N))
            for target in range(0, population2.N):
                if p[source * population2.N + target] < probability:
                    connections_source.append(source)
                    connections_target.append(target)

        connections = np.zeros([len(connections_target), 2])
        connections[:, 0] = np.asarray(connections_source)
        connections[:, 1] = np.asarray(connections_target)

        print ('Saving connection matrix: semd_rcn')
        np.save(save_path + 'semd_rcn_v%i' % (i + 1), connections)
    return connections[:,0], connections[:,1]












































































def gabor_on(population1, population2, save_path, save):
    orientation = [0, 45, 90, 135]
    filename1 = save_path + '%s_%s_%s_%s' %('Gabor_on', str(int(sqrt(population1.N))), str(int(sqrt(population1.N))), str(orientation[0])) + '.npy'
    filename2 = save_path + '%s_%s_%s_%s' %('Gabor_on', str(int(sqrt(population1.N))), str(int(sqrt(population1.N))), str(orientation[1])) + '.npy'
    filename3 = save_path + '%s_%s_%s_%s' %('Gabor_on', str(int(sqrt(population1.N))), str(int(sqrt(population1.N))), str(orientation[2])) + '.npy'
    filename4 = save_path + '%s_%s_%s_%s' %('Gabor_on', str(int(sqrt(population1.N))), str(int(sqrt(population1.N))), str(orientation[3])) + '.npy'

    start = np.floor(np.sqrt(population1.N)/2).astype(int)
    stepwidth = np.sqrt(population1.N).astype(int)
    offset = ((np.floor(np.sqrt(population1.N)/2)-1)/2).astype(int)

    target = 0
    connections_source = []
    connections_target = []

    for base in range(start*stepwidth, start*stepwidth+stepwidth, 1):
        for offset_tmp in range(-(offset*stepwidth), (offset*stepwidth)+1, stepwidth):
            source = base + offset_tmp
            connections_source.append(source)
            connections_target.append(target)

    connection_0 = np.zeros([len(connections_source), 2])
    connection_0[:, 0] = np.asarray(connections_source)
    connection_0[:, 1] = np.asarray(connections_target)

    target = 1
    offset_x = 0
    connections_source = []
    connections_target = []

    for base in range(0, population1.N, stepwidth):
        for offset_tmp in range(-offset, offset+1):
            source = base + offset_x + offset_tmp
            if source < base or source > base+stepwidth-1:
                continue
            connections_source.append(source)
            connections_target.append(target)

        offset_x += 1
    connection_45 = np.zeros([len(connections_source), 2])
    connection_45[:, 0] = np.asarray(connections_source)
    connection_45[:, 1] = np.asarray(connections_target)


    target = 2
    

    for base in range(start, population1.N-start, stepwidth):
        for offset_tmp in range(-offset, offset+1):
            source = base + offset_tmp
            connections_source.append(source)
            connections_target.append(target)
    connection_90 = np.zeros([len(connections_source), 2])
    connection_90[:, 0] = np.asarray(connections_source)
    connection_90[:, 1] = np.asarray(connections_target)


    target = 3
    connections_source = []
    connections_target = []
    offset_x = 0

    for base in range(stepwidth-1, population1.N, stepwidth):
        for offset_tmp in range(-offset, offset+1):
            source = base + offset_tmp - offset_x
            if source > base or source < base-stepwidth+1:
                continue
            connections_source.append(source)
            connections_target.append(target)
        offset_x += 1

    connection_135 = np.zeros([len(connections_source), 2])
    connection_135[:, 0] = np.asarray(connections_source)
    connection_135[:, 1] = np.asarray(connections_target)


    if save:
        print ('Saving connection matrix:')
        np.save(filename1, connection_0)
        np.save(filename2, connection_45)
        np.save(filename3, connection_90)
        np.save(filename4, connection_135)

    return connection_0, connection_45, connection_90, connection_135



def gabor_off(population1, population2, save_path, save):
    orientation = [0, 45, 90, 135]
    filename1 = save_path + '%s_%s_%s_%s' %('Gabor_off', str(int(sqrt(population1.N))), str(int(sqrt(population1.N))), str(orientation[0])) + '.npy'
    filename2 = save_path + '%s_%s_%s_%s' %('Gabor_off', str(int(sqrt(population1.N))), str(int(sqrt(population1.N))), str(orientation[1])) + '.npy'
    filename3 = save_path + '%s_%s_%s_%s' %('Gabor_off', str(int(sqrt(population1.N))), str(int(sqrt(population1.N))), str(orientation[2])) + '.npy'
    filename4 = save_path + '%s_%s_%s_%s' %('Gabor_off', str(int(sqrt(population1.N))), str(int(sqrt(population1.N))), str(orientation[3])) + '.npy'

    start = np.floor(np.sqrt(population1.N)/2).astype(int)
    stepwidth = np.sqrt(population1.N).astype(int)
    offset = ((np.floor(np.sqrt(population1.N)/2)-1)/2).astype(int)

    target = 0
    connections_source = []
    connections_target = []

    for base in range(start*stepwidth, start*stepwidth+stepwidth, 1):
        if not offset == 0:
            for offset_tmp in range(-(offset*stepwidth)*2, (offset*stepwidth)*2+1, stepwidth):
                if not offset_tmp in range(-(offset*stepwidth), (offset*stepwidth)+1, stepwidth):
                    source = base + offset_tmp
                    connections_source.append(source)
                    connections_target.append(target)
        else:
            for offset_tmp in range(-5, 6, 10):
                source = base + offset_tmp
                connections_source.append(source)
                connections_target.append(target)

    connection_0 = np.zeros([len(connections_source), 2])
    connection_0[:, 0] = np.asarray(connections_source)
    connection_0[:, 1] = np.asarray(connections_target)

    target = 1
    offset_x = 0
    connections_source = []
    connections_target = []

    for base in range(0, population1.N, stepwidth):
        if not offset == 0:
            for offset_tmp in range(-offset*2, offset*2+1):
                if not offset_tmp in range(-offset, offset+1):
                    source = base + offset_x + offset_tmp
                    if source < base or source > base+stepwidth-1:
                        continue
                    connections_source.append(source)
                    connections_target.append(target)
        else:
            for offset_tmp in range(-1,2,2):
                source = base + offset_x + offset_tmp
                if source < base or source > base+stepwidth-1:
                    continue
                connections_source.append(source)
                connections_target.append(target)   

        offset_x += 1
    connection_45 = np.zeros([len(connections_source), 2])
    connection_45[:, 0] = np.asarray(connections_source)
    connection_45[:, 1] = np.asarray(connections_target)


    target = 2

    for base in range(start, population1.N-start, stepwidth):
        if not offset == 0:
            for offset_tmp in range(-offset*2, offset*2+1):
                if not offset_tmp in range(-offset, offset+1):
                    source = base + offset_tmp
                    connections_source.append(source)
                    connections_target.append(target)
        else:
            for offset_tmp in range(-1,2,2):
                source = base + offset_tmp
                connections_source.append(source)
                connections_target.append(target)

    connection_90 = np.zeros([len(connections_source), 2])
    connection_90[:, 0] = np.asarray(connections_source)
    connection_90[:, 1] = np.asarray(connections_target)
    

    target = 3
    connections_source = []
    connections_target = []
    offset_x = 0

    for base in range(stepwidth-1, population1.N, stepwidth):
        if not offset == 0:
            for offset_tmp in range(-offset*2, offset*2+1):
                if not offset_tmp in range(-offset, offset+1):
                    source = base + offset_tmp - offset_x
                    if source > base or source < base-stepwidth+1:
                        continue
                    connections_source.append(source)
                    connections_target.append(target)
        else:
            for offset_tmp in range(-1,2,2):
                source = base + offset_tmp - offset_x
                if source > base or source < base-stepwidth+1:
                    continue
                connections_source.append(source)
                connections_target.append(target)
        offset_x += 1

    connection_135 = np.zeros([len(connections_source), 2])
    connection_135[:, 0] = np.asarray(connections_source)
    connection_135[:, 1] = np.asarray(connections_target)


    if save:
        print ('Saving connection matrix:')
        np.save(filename1, connection_0)
        np.save(filename2, connection_45)
        np.save(filename3, connection_90)
        np.save(filename4, connection_135)

    return connection_0, connection_45, connection_90, connection_135