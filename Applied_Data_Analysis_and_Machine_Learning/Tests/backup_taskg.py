
def perform_OLS_regression(n_points=300, degrees=10, r_seed=79, noisy=True, scaling=True): 
    np.random.seed(r_seed)
    
    

    MSE_train_list = np.empty(degrees)
    MSE_test_list = np.empty(degrees)
    R2_train_list = np.empty(degrees)
    R2_test_list = np.empty(degrees)
    betas_list = []
    preds_cn = []
    X = create_X(x,y,degrees, centering=scaling)

    for degree in range(1, degrees+1):
        
        x_train, x_test, z_train, z_test = train_test_split(X[:, :int((degree+1)*(degree+2)/2)], z.ravel(), test_size=0.2)

        if scaling:
            x_train_mean = np.mean(x_train, axis=0) 
            z_train_mean = np.mean(z_train, axis=0)  
            x_train -= x_train_mean
            x_test -= x_train_mean
            z_train_centered = z_train - z_train_mean
        else: 
            z_train_mean = 0
            z_train_centered = z_train            

        beta_SVD_cn = compute_optimal_parameters2(x_train, z_train_centered)
        betas_list.append(beta_SVD_cn)
        # Shifted intercept for use when data is not centered
        #intercept = np.mean(z_train_mean - x_train_mean @ beta_SVD_cn)
        
        preds_visualization_cn = predict(X[:, :int((degree+1)*(degree+2)/2)], beta_SVD_cn, z_train_mean)
        preds_visualization_cn = preds_visualization_cn.reshape(n_points, n_points)
        preds_cn.append(preds_visualization_cn)

        preds_train_cn = predict(x_train, beta_SVD_cn, z_train_mean)
        preds_test_cn = predict(x_test, beta_SVD_cn, z_train_mean)

        MSE_train_list[degree-1] = MSE(z_train, preds_train_cn)
        MSE_test_list[degree-1] = MSE(z_test, preds_test_cn)
        R2_train_list[degree-1] = R2(z_train, preds_train_cn)
        R2_test_list[degree-1] = R2(z_test, preds_test_cn)

    return betas_list, MSE_train_list, MSE_test_list, R2_train_list, R2_test_list,  preds_cn, x, y, z

(betas, MSE_train, MSE_test, 
R2_train, R2_test, preds_cn, x, y, z) = perform_OLS_regression(scaling=False, degrees=20, r_seed=79)
