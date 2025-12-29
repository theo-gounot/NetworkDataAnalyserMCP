import numpy as np
from scipy.stats import betabinom
import time

verbose = False

# Compute the log-likelihood value for the normal distribution
def loglik(x, loc, scale):
    n = len(x)
    c = 1/np.sqrt(2*np.pi)
    y = n*np.log(c/scale) -(1/(2*scale**2))*((x-loc)**2).sum()
    return y

# Voting Windows Changepoint Detection
def vwcd(X, w=20, w0=20, ab=1, p_thr=0.8, vote_p_thr=0.9, vote_n_thr=0.5, y0=0.5, yw=0.9, aggreg='mean'):
    """
    Voting Windows Changepoint Detection
   
    Parameters:
    ----------
    X (numpy array): the input time-series
    w (int): sliding window size
    w0 (int): pre-change estimating window size
    ab (int): Beta-binomial alpha and beta hyperp - prior dist. window
    p_thr (float): threshold probability to an window decide for a changepoint
    vote_p_thr (float): threshold probabilty to decide for a changepoint after aggregation
    vote_n_thr (float): min. number of votes to decide for a changepoint (fraction of w)
    y0 (float): Logistic prior hyperparameter
    yw (float): Logistic prior hyperparameter
    aggreg (str): aggregation function ('mean' or 'posterior')

    Returns:
    -------
    CP (list): change-points
    M0 (list): estimated mean of the segments
    S0 (list): estimated standar deviation of the segments
    elapsedTime (float): running-time  (microseconds)
    """
    
    # Auxiliary functions
    # Compute the window posterior probability given the log-likelihood and prior
    # using the log-sum-exp trick
    def pos_fun(ll, prior, tau):
        c = np.nanmax(ll)
        lse = c + np.log(np.nansum(prior*np.exp(ll - c)))
        p = ll[tau] + np.log(prior[tau]) - lse
        return np.exp(p)

    # Aggregate a list of votes - compute the posterior probability
    def votes_pos(vote_list, prior_v):
        vote_list = np.array(vote_list)
        prod1 = vote_list.prod()*prior_v
        prod2 = (1-vote_list).prod()*(1-prior_v)
        p = prod1/(prod1+prod2)
        return p

    # Prior probabily for votes aggregation
    def logistic_prior(x, w, y0, yw):
        a = np.log((1-y0)/y0)
        b = np.log((1-yw)/yw)
        k = (a-b)/w
        x0 = a/k
        y = 1./(1+np.exp(-k*(x-x0)))
        return y
    
    # Auxiliary variables
    N = len(X)
    vote_n_thr_val = np.floor(w*vote_n_thr)

    # Prior probatilty for a changepoint in a window - Beta-B
    i_ = np.arange(0,w-3)
    prior_w = betabinom(n=w-4,a=ab,b=ab).pmf(i_)

    # prior for vot aggregation
    x_votes = np.arange(1,w+1)
    prior_v = logistic_prior(x_votes, w, y0, yw) 

    votes = {i:[] for i in range(N)} # dictionary of votes 
    votes_agg = {}  # aggregated voteylims

    lcp = 0 # last changepoint
    CP = [] # changepoint list
    M0 = [] # list of post-change mean
    S0 = [] # list of post-change standard deviation

    startTime = time.time()
    for n in range(N):
        if n>=w-1:
            
            # estimate the paramaters (w0 window)
            if n == lcp+w0:
                # estimate the post-change mean and variace
                m_w0 = X[n-w0+1:n+1].mean()
                s_w0 = X[n-w0+1:n+1].std(ddof=1)
                M0.append(m_w0)
                S0.append(s_w0)
            
            # current window
            Xw = X[n-w+1:n+1]
            
            LLR_h = []
            for nu in range(1,w-3+1):
            #for nu in range(w):
                # MLE and log-likelihood for H1
                x1 = Xw[:nu+1] #Xw até nu
                m1 = x1.mean()
                s1 = x1.std(ddof=1)
                if np.round(s1,3) == 0:
                    s1 = 0.001
                logL1 = loglik(x1, loc=m1, scale=s1)
                
                # MLE and log-likelihood  for H2
                x2 = Xw[nu+1:]
                m2 = x2.mean()
                s2 = x2.std(ddof=1)
                if np.round(s2,3) == 0:
                    s2 = 0.001
                logL2 = loglik(x2, loc=m2, scale=s2)

                # log-likelihood ratio
                llr = logL1+logL2
                LLR_h.append(llr)

            
            # Compute the posterior probability
            LLR_h = np.array(LLR_h)
            pos = [pos_fun(LLR_h, prior_w, nu) for nu in range(w-3)]
            pos = [np.nan] + pos + [np.nan]*2
            pos = np.array(pos)
            
            # Compute the MAP (vote)
            p_vote_h = np.nanmax(pos)
            nu_map_h = np.nanargmax(pos)
            
            # Store the vote if it meets the hypothesis test threshold
            if p_vote_h >= p_thr:
                j = n-w+1+nu_map_h # Adjusted index 
                votes[j].append(p_vote_h)
            
            # Aggregate the votes for X[n-w+1]
            votes_list = votes[n-w+1]
            num_votes = len(votes_list)
            if num_votes >= vote_n_thr_val:
                if aggreg == 'posterior':
                    agg_vote = votes_pos(votes_list, prior_v[num_votes-1])
                elif aggreg == 'mean':
                    agg_vote = np.mean(votes_list)
                votes_agg[n-w+1] = agg_vote
                
                # Decide for a changepoit
                if agg_vote > vote_p_thr:
                    if verbose: print(f'Changepoint at n={n-w+1}, p={agg_vote}, n={num_votes} votes')
                    lcp = n-w+1 # last changepoint
                    CP.append(lcp)

    endTime = time.time()
    elapsedTime = endTime-startTime
    
    # If the last segment wasn't added to M0/S0 (because n < lcp+w0 for the last part)
    # The original code might not handle the very last segment mean if it is shorter than w0
    # But let's check: M0.append(m_w0) happens when n == lcp+w0
    # If the series ends before that, the last segment stats are not recorded?
    # I should probably add the last segment stats manually if they are missing
    
    if len(CP) + 1 > len(M0):
        # Calculate stats for the last segment
        last_cp = CP[-1] if CP else 0
        seg = X[last_cp:]
        if len(seg) > 0:
            M0.append(seg.mean())
            S0.append(seg.std(ddof=1) if len(seg) > 1 else 0.0)

    # Note: The original code logic for M0/S0 seems to only record "stable" estimates
    # after w0 samples. I might need to refine this for the "brief statistical description"
    # requirement, but let's stick to the ported logic and maybe augment it in the wrapper.

    return CP, M0, S0, elapsedTime

def get_segments(X, CP):
    """
    Split the time-series into segments based on change-points and compute stats.
    """
    segments = []
    change_points = [-1] + CP + [len(X)-1]
    
    for i in range(len(change_points) - 1):
        s = int(change_points[i] + 1)
        e = int(change_points[i+1])
        if s > e: continue 
        
        segment_data = X[s:e+1]
        mean_val = float(np.mean(segment_data))
        std_val = float(np.std(segment_data, ddof=1)) if len(segment_data) > 1 else 0.0
        
        segments.append({
            "segment_index": i,
            "start_index": s,
            "end_index": e,
            "length": len(segment_data),
            "mean": mean_val,
            "std": std_val,
            "min": float(np.min(segment_data)),
            "max": float(np.max(segment_data)),
            "median": float(np.median(segment_data))
        })
    return segments
