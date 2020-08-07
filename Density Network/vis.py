import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colorbar
import matplotlib.colors
from matplotlib.pyplot import cm
#from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os


def vis2d(SEinput, VDinput, VDtarget, VDpred, k, path):
    vdtarget, vdpred, seinput, vdinput = [], [], [], []
    semin, semax = [], []
    for i in range(VDtarget.shape[0]):
        for j in range(VDtarget.shape[1]):
            vdtarget_temp = VDtarget[i, :, :, j] 
            vdpred_temp = VDpred[i, :, :, j] 
            seinput_temp = SEinput[i, :, :, j]
            vdinput_temp = VDinput[i, :, :, j]

            vdtarget.append(vdtarget_temp)
            vdpred.append(vdpred_temp)
            seinput.append(seinput_temp)
            vdinput.append(vdinput_temp)

            semin.append(np.amin(SEinput[i]))
            semax.append(np.amax(SEinput[i]))
            

    vdtarget = np.array(vdtarget)
    vdpred = np.array(vdpred)
    seinput = np.array(seinput)
    vdinput = np.array(vdinput)
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(0, vdtarget.shape[0], 2):
        cols = ['{}'.format(name) for name in ['CurrentComp  ', '  CurrentVD', '  TargetVD ', ' PredVD']]
        rows = ['Slice{}'.format(num) for num in [i, i+1]]
        
        fig, axes = plt.subplots(2,4)#, constrained_layout=True)
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)
            
        for ax, row in zip(axes[:,0], rows):
            ax.set_ylabel(row, rotation='vertical')
        for axx, axy in zip(axes[0], axes[1]):
            axx.set_yticklabels([])
            axx.set_xticklabels([])
            axy.set_yticklabels([])
            axy.set_xticklabels([])
            pass
        for j in range(2):
            im0 = axes[j,0].imshow(seinput[i+j, :, :, 0], vmin=semin[i+j], vmax=semax[i+j], cmap='jet')
            cbar0 = fig.colorbar(im0, ax=axes[j,0], orientation='horizontal', pad=0.05)#, ticks=[0.0, 1.0])
            # cbar0.ax.set_xticklabels([ '0.0', '1.0'])

            im1 = axes[j,1].imshow(vdinput[i+j, :, :, 0], vmin=0.0, vmax=1.0, cmap='jet')
            cbar1 = fig.colorbar(im1, ax=axes[j,1], orientation='horizontal', pad=0.05, ticks=[0.0, 1.0])
            cbar1.ax.set_xticklabels([ '0.0', '1.0'])

            im2 = axes[j,2].imshow(vdtarget[i+j, :, :, 0], vmin=0.0, vmax=1.0, cmap='jet')
            cbar2 = fig.colorbar(im2, ax=axes[j,2], orientation='horizontal', pad=0.05, ticks=[0.0, 1.0])
            cbar2.ax.set_xticklabels([ '0.0', '1.0'])

            im3 = axes[j,3].imshow(vdpred[i+j, :, :, 0], vmin=0.0, vmax=1.0, cmap='jet')
            cbar3 = fig.colorbar(im3, ax=axes[j,3], orientation='horizontal', pad=0.05, ticks=[0.0, 1.0])
            cbar3.ax.set_xticklabels([ '0.0', '1.0'])

        #plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(path,'%s_%s.png'%(k, i)))
        plt.close()


def vis3d(Input, Inouts, Target, Predicted, testId, path='MPLs'):
    # k = 0
    for i in range(Inouts.shape[0]):
        #for i in range(Inouts.shape[0]):
            #if np.sum(Inouts[j,:,:,i]) > 50.0:
        dataId = str(testId) + '_' + str(i)
        VisMPLComparison(Input[i,:,:,:,0], Inouts[i,:,:,:,0], Target[i,:,:,:,0], Predicted[i,:,:,:,0], dataId, path)
        #k = k + 1

def VisMPLComparison(Input, Inouts, Target, Predicted, DataId, path='MPLs'):
    fig = plt.figure(figsize=plt.figaspect(0.25))
    axInput = fig.add_axes([0.02, 0.1, 0.25, 0.8], projection='3d')
    axCbInput = fig.add_axes([0.25, 0.3, 0.015, 0.45])
    axTarget = fig.add_axes([0.3, 0.1, 0.25, 0.8], projection='3d')
    axCbTarget = fig.add_axes([0.6, 0.3, 0.015, 0.45])
    axPredicted = fig.add_axes([0.65, 0.1, 0.25, 0.8], projection='3d')
    axCbPredicted = fig.add_axes([0.9, 0.3, 0.015, 0.45])

    InputNorm = matplotlib.colors.Normalize(vmin = np.amin(Input), vmax = np.amax(Input))
    TargetNorm = matplotlib.colors.Normalize(vmin = 0.0, vmax = 1.0)
    PredictedNorm = matplotlib.colors.Normalize(vmin = 0.0, vmax = 1.0)
    def _Inputcolors(i,j,k):
        if Inouts[i,j,k] == 0:
            return (0.0, 0.0, 0.0, 0.0)
        else:
            color = matplotlib.cm.ScalarMappable(norm=InputNorm, cmap = plt.cm.jet).to_rgba(Input[i,j,k])
            return color
    def _Targetcolors(i,j,k):
        if Inouts[i,j,k] == 0:
            return (0.0, 0.0, 0.0, 0.0)
        else:
            color = matplotlib.cm.ScalarMappable(norm=TargetNorm, cmap = plt.cm.jet).to_rgba(Target[i,j,k])
            return color
    def _Predictedcolors(i,j,k):
        if Inouts[i,j,k] == 0:
            return (0.0, 0.0, 0.0, 0.0)
        else:
            color = matplotlib.cm.ScalarMappable(norm=PredictedNorm, cmap = plt.cm.jet).to_rgba(Predicted[i,j,k])
            return color
    Inputcolors = lambda i,j,k : _Inputcolors(i,j,k)
    Targetcolors = lambda i,j,k : _Targetcolors(i,j,k)
    Predictedcolors = lambda i,j,k : _Predictedcolors(i,j,k)
    shape = Inouts.shape
    facecolorsInput = np.array([[[Inputcolors(x,y,z) for z in range(shape[2])] for y in range(shape[1])] for x in range(shape[0])])
    facecolorsTarget = np.array([[[Targetcolors(x,y,z) for z in range(shape[2])] for y in range(shape[1])] for x in range(shape[0])])
    facecolorsPredicted = np.array([[[Predictedcolors(x,y,z) for z in range(shape[2])] for y in range(shape[1])] for x in range(shape[0])])
    axInput.voxels(Inouts, facecolors=facecolorsInput)
    axInput.text2D(0.5, 1.0, "CurrentCompliance", transform=axInput.transAxes)
    cbar1 = matplotlib.colorbar.ColorbarBase(axCbInput, cmap=plt.cm.jet, norm=InputNorm, orientation='vertical')  
    axTarget.voxels(Inouts, facecolors=facecolorsTarget)
    axTarget.text2D(0.5, 1.0, "TargetDensity", transform=axTarget.transAxes)
    cbar2 = matplotlib.colorbar.ColorbarBase(axCbTarget, cmap=plt.cm.jet, norm=TargetNorm, orientation='vertical')  
    axPredicted.voxels(Inouts, facecolors=facecolorsPredicted)
    axPredicted.text2D(0.5, 1.0, "PredictedDensity", transform=axPredicted.transAxes)
    cbar3 = matplotlib.colorbar.ColorbarBase(axCbPredicted, cmap=plt.cm.jet, norm=PredictedNorm, orientation='vertical')  
    #fig.text(.5, .05, '%s is the target volume fraction'%VolFrac, ha='center')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path,'%s.png'%(DataId)))
    plt.close()


