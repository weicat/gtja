from numba import njit
import numpy as np



#%%
#需要所有东西都align 过

@njit
def getAvailableVolume(prev_arr, now_arr, adjclose, mv,
                       st, suspend, zt, dt):
    
    '''
    设置exclude的目的是为了让退市的股票保持weight =0 走掉
    weight = 0 在之前weight 转 volume的时候已经筛选过了
    '''
    # tt = (now_arr * adjclose).sum()
    
    exclude = set(np.where(adjclose == 0)[0])
    else_index = set([int(i) for i in range(0)])
    if not st is None:
        st_index = np.where(st == 1)[0]
        now_arr[st_index] = 0
        else_index = else_index.union(set(st_index))
    
    if not suspend is None:
        suspend_index = np.where(suspend == 1)[0]
        suspend_index = np.array(list(set(suspend_index) - exclude))
        now_arr[suspend_index] = prev_arr[suspend_index]
        else_index = else_index.union(set(suspend_index))

    if not zt is None:
        zt_index = np.where(zt == 1)[0]
        zt_index = np.array(list(set(zt_index) - exclude))
        temp_now = now_arr[zt_index]
        temp_prev = prev_arr[zt_index]
        p_index = np.where(temp_now > temp_prev)
        temp_now[p_index] = temp_prev[p_index]
        now_arr[zt_index] = temp_now
        else_index = else_index.union(set(zt_index))


    if not dt is None:
        dt_index = np.where(dt == 1)[0]
        dt_index = np.array(list(set(dt_index) - exclude))
        temp_now = now_arr[dt_index]
        temp_prev = prev_arr[dt_index]
        p_index = np.where(temp_now < temp_prev)
        temp_now[p_index] = temp_prev[p_index]
        now_arr[dt_index] = temp_now
        else_index = else_index.union(set(dt_index))
    
    all_index = set(list(range(len(now_arr))))
    good_index = np.array(list(all_index - else_index))
    else_index = np.array(list(else_index))
    
    
    
    dollar_arr = now_arr * adjclose
    if len(else_index) == 0:
        unmoved_mv = 0
    else:
        unmoved_mv = dollar_arr[else_index].sum()
    target_mv = mv - unmoved_mv
    if len(good_index) == 0:
        exist_mv = 0
    else:
        exist_mv = dollar_arr[good_index].sum()
    
    if exist_mv == 0:
        money = target_mv
    else:
        now_arr[good_index] = now_arr[good_index]/exist_mv * target_mv
        money = 0
    
    
    
    # tf = [np.isnan(i) for i in now_arr]    
    # tf = np.array([1 if i== True else 0 for i in tf])
    # if np.sum(tf)!= 0 :
    #     print(mv, unmoved_mv, dollar_arr.sum(),tt )
    #     raise ValueError('a')
    return now_arr, money


#%%
@njit
def getMV(volume_arr, adjclose, money):
    return (volume_arr * adjclose).sum() + money 


@njit 
def getVolumeFromWeight(weight_arr, adjclose, mv):
    weight_arr[adjclose ==0 ] = 0
    weight_arr = weight_arr/weight_arr.sum()
    dollar_arr = weight_arr * mv
    filled = dollar_arr[dollar_arr > 0]/adjclose[dollar_arr > 0]
    volume_arr = np.zeros(shape = weight_arr.shape)
    volume_arr[dollar_arr > 0] = filled    
    return volume_arr
    


    


@njit 
def getPortfolioReturn(weight_arrs,
                       balance_index,
                       adjcloses, 
                       sts, suspends, zts, dts):
    
    mv_lst = np.zeros(adjcloses.shape[0])
    actual_volume_lst = np.zeros(adjcloses.shape)
    prev_volume_arr = np.zeros(adjcloses.shape[1])
    
    balance_count = 0
    money = 0
    
    for i in range(adjcloses.shape[0]):
        
        adjclose = adjcloses[i,:]
        if i == 0:
            mv = 1.0
        else:    
            mv = getMV(prev_volume_arr, adjclose, money)
        
        mv_lst[i] = mv
        if sts is None:
            st = None
        else:
            st = sts[i,:]
        if suspends is None:
            suspend = None
        else:
            suspend = suspends[i,:]
        if zts is None:
            zt = None
        else:
            zt = zts[i,:]
        if dts is None:
            dt = None
        else:
            dt = dts[i,:]
        
        if i in balance_index:
            now_weight_arr = weight_arrs[balance_index[balance_count],:]
            now_volume_arr = getVolumeFromWeight(now_weight_arr, adjclose, mv)
            prev_volume_arr, money = getAvailableVolume(prev_volume_arr, now_volume_arr,
                                                 adjclose, mv, 
                                                 st, suspend, zt, dt)
            actual_volume_lst[i,:] = prev_volume_arr
            balance_count += 1
        else:   
            now_volume_arr = prev_volume_arr    
            prev_volume_arr, money = getAvailableVolume(prev_volume_arr, now_volume_arr,
                                                 adjclose, mv, 
                                                 st, suspend, zt, dt)
            actual_volume_lst[i,:] = prev_volume_arr
            

        
        
        
    return mv_lst, actual_volume_lst
