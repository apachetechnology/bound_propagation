from matplotlib import pyplot as plt

def getNormalImg(aImg, listStd, listMean):
    return aImg * listStd[:, None, None] + listMean[:, None, None]

def ShowImg(img, strTitle, strName):
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.title(strTitle)
    if strName != None:
        plt.savefig(strName, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def ShowImg(img, strTitle, bTransform, 
            nInCh, nInDim, cDPI = 80, strName=None):
    
    figsize = nInDim / float(cDPI), nInDim / float(cDPI)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    # Hide spines, ticks, etc.
    ax.axis('off')

    if bTransform:
        img = img / 2 + 0.5  # unnormalize
        
    #npimg = img.numpy()  # convert from tensor
    #cv2.imwrite('./local-results/test.jpg', img.permute(1, 2, 0).numpy()*255)
    if nInCh == 1:
        ax.imshow(img.permute(1, 2, 0), cmap='gray')
    else:
        ax.imshow(img.permute(1, 2, 0))
    
    plt.title(strTitle)
    if strName != None:
        plt.savefig(strName, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()