import cv2


def get_histogram_match(target, previous_histograms, best_histograms, prior):
    result = 1
    target_hist = get_histogram(target)
    prior_hist = get_histogram(prior)
    if previous_histograms.qsize() > 0:
        result = compare_histograms(target_hist, previous_histograms, best_histograms, prior_hist)
    if (result < 0.5 and result != -1) or best_histograms.qsize() == 0:
        update_histogram(target_hist, best_histograms)
    update_histogram(target_hist, previous_histograms)
    return result


def update_histogram(target, histograms):
    if histograms.qsize() > 10:
        histograms.get()
    histograms.put(target)

def get_histogram(image):
   #images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def compare_histograms(target, previous_histograms, best_histograms, prior):
    method = cv2.HISTCMP_CHISQR
    method2 = cv2.HISTCMP_BHATTACHARYYA
    target_result = 0
    prior_result = 0
    for histogram in list(best_histograms.queue):
        target_result += cv2.compareHist(target, histogram, method)
        prior_result += cv2.compareHist(prior, histogram, method)
    target_result = target_result / best_histograms.qsize()
    prior_result = prior_result / best_histograms.qsize()
    if (target_result / prior_result > 1.2):
        return -1
    return target_result / prior_result
