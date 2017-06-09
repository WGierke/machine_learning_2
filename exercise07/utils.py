def analyze_worstcase_performance(text2bow,kernel):

    # Extract the two longest texts in the dataset
    texts = list(open('newsgroup-data/newsgroup-train-data.txt','r'))
    sortedtexts = sorted(texts,key=len)[::-1]

    longtext1 = sortedtexts[0]
    longtext2 = sortedtexts[1]

    # Build their bag-of-words representation
    bow1 = text2bow(longtext1)
    bow2 = text2bow(longtext2)

    # Apply the kernel function and compute its output and running time
    import time
    start = time.time(); output = kernel(bow1,bow2); end = time.time()
    print('kernel score: %.3f , computation time: %.3f'%(output,end-start))

