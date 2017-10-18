import numpy as np
import math

def get_quantiles(preds, num_quantiles=10):
    """
        Calculates the num_quantiles from the predictions for each class
    """
    k = np.linspace(0, 100, num=num_quantiles)
    ret = np.percentile(preds, k, axis=0)
    return ret


def argmax_rand(a):
    # own argmax that takes a random argmax if there are multiple maxes in the array
    return int(np.random.choice(np.where(a == np.max(a))[0], 1))


def get_MAPS(pred):
    """
    Args:
        pred a [Number of Examples, Number of classes Tensor]

    Returns:
        The MAPs for all classes
    """
    num_classes = np.shape(pred)[1]
    maps = np.empty(0)
    for j in range(0, num_classes):
        test = np.histogram(pred[:, j], range=[0, 1], bins=31)
        rel_haef = test[0]
        # Taking all obversations between the borders of the MAP bin
        max_bin = argmax_rand(test[0])
        lb = test[1][max_bin]
        rb = test[1][max_bin + 1]
        pred_c = pred[:, j]
        obs_sel = pred_c[lb <= pred_c]
        obs_sel = obs_sel[obs_sel <= rb]
        #         if (len(obs_sel) == 0): #Exapand the boundates
        #             lb = test[1][max_bin] - 1e-6
        #             rb = test[1][max_bin + 1] + 1e-6
        #             obs_sel = d[lb <= d[:,j],j]
        #             obs_sel = obs_sel[obs_sel <= rb]
        map = np.median(obs_sel)
        maps = np.append(maps, map)
    return (maps)


def get_VI_MAPS(preds, pmass=0.66):
    """
        Agrs:
            preds a 2 dimensional array
            pmass mass of the credibility interval
        Returns:
            the lower CI, MAPS, Upper CI, index of predicted class (max MAP)
    """
    num_classes = np.shape(preds)[1]
    MAPS = get_MAPS(preds)
    Pred = np.argmax(MAPS)
    MAP = MAPS[Pred]
    if (math.isnan(MAP)):
        print(" Pred", Pred, " MAP ", MAP, " MAPS ", MAPS)
        import sys
        sys.exit("Dumm")
    VI = np.zeros(2)
    MR = np.sum(preds[:, Pred] > (MAP)) / len(preds)  # Mass right from MAP
    ML = np.sum(preds[:, Pred] <= (MAP)) / len(preds)

    # print("  ML ", ML, "  MR ", MR, " Pred", Pred, " MAP ", MAP)
    if MR < pmass / 2:
        # Right not possible
        # print("Right not possible ", MR, " ", MAP, " max pred", Pred)
        UP = 1.0
        LO = np.percentile(preds[:, Pred], 100 * (1 - pmass))
    elif ML < pmass / 2:
        # print("Left not possible ML ", ML)
        UP = np.percentile(preds[:, Pred], 100 * pmass)
        LO = 0.0
    else:
        # print("All possible MR ", MR, "  ", ML)
        LO = np.percentile(preds[:, Pred], 100 * ((1 - pmass) / 2))
        UP = np.percentile(preds[:, Pred], 100 * (1 - ((1 - pmass) / 2)))
    return LO, MAP, UP, Pred