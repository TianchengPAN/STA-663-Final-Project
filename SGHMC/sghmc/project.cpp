<%
cfg['compiler_args'] = ['-std=c++11']
cfg['include_dirs'] = ['eigen']
setup_pybind11(cfg)
%>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <stdexcept>
#include <algorithm> // std::random_shuffle
#include <random>

#include <Eigen/LU>
#include <Eigen/Dense>
#include <eigen/Eigen/Core>

namespace py = pybind11;
using std::default_random_engine;
using std::normal_distribution;
        
default_random_engine re{100};
normal_distribution<double> norm(0, 1);
auto rnorm = bind(norm, re);

Eigen::MatrixXd rnorm_vec(int n) {
    Eigen::MatrixXd res_vec = Eigen::MatrixXd::Zero(n, 1);
    for (int i=0; i<n; i++) {
        res_vec(i,0) = rnorm();
    }
    return res_vec;
}
    
Eigen::MatrixXd gradU_noisyFig1(Eigen::MatrixXd theta) {
    Eigen::MatrixXd xs = -4*theta.array() + 4*theta.array().pow(3) + 2*rnorm_vec(theta.rows()).array();
    return xs;
}
     
Eigen::MatrixXd gradU_mixNormals(Eigen::MatrixXd theta, Eigen::MatrixXd x, int n, int batch_size) {
    int p = theta.rows();
    Eigen::ArrayXd c_0 = theta(0,0) - x.array();
    Eigen::ArrayXd c_1 = theta(1,0) - x.array();
    Eigen::ArrayXd star = 0.5 * (-0.5 * c_0.pow(2)).exp() + 0.5 * (-0.5 * c_1.pow(2)).exp();
    Eigen::ArrayXd star_prime;
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(p, 1);
    for (int i=0; i<p; i++) {
        star_prime = 0.5 * (-0.5 * (theta(i,0) - x.array()).pow(2)).exp() * (theta(i,0) - x.array());
        grad(i,0) = -theta(i,0)/10 - (n/batch_size)*(star_prime/star).sum();
    }
    return grad;
} 

Eigen::MatrixXd sghmc(std::string gradU_choice, Eigen::MatrixXd eta, int niter, Eigen::MatrixXd alpha, Eigen::MatrixXd theta_0, Eigen::MatrixXd V_hat, Eigen::MatrixXd dat, int batch_size){
    int p = theta_0.rows();
    int n = dat.rows();     
    int p_dat = dat.cols();
    int nbatches = n / batch_size; 
    Eigen::MatrixXd dat_temp = dat;
    Eigen::MatrixXd dat_batch = Eigen::MatrixXd::Zero(batch_size, p_dat);
    Eigen::MatrixXd gradU_batch = Eigen::MatrixXd::Zero(p, 1);
    Eigen::MatrixXd theta_samps = Eigen::MatrixXd::Zero(p, niter*(n/batch_size));
    std::vector<int> ind;
    Eigen::MatrixXd beta_hat = 0.5 * V_hat * eta;
    Eigen::MatrixXd Sigma = 2.0 * (alpha - beta_hat) * eta;
    Eigen::LLT<Eigen::MatrixXd> lltOfSig(Sigma); 
    if(lltOfSig.info() == Eigen::NumericalIssue){ 
        return theta_samps; 
    }
    Eigen::MatrixXd Sig_chol = lltOfSig.matrixL(); 
    if(batch_size > n){ 
        return theta_samps; 
    }
    Eigen::LLT<Eigen::MatrixXd> lltOfeta(eta); 
    Eigen::MatrixXd eta_chol = lltOfeta.matrixL(); 
    Eigen::MatrixXd nu = eta_chol * rnorm_vec(p); 
    Eigen::MatrixXd theta = theta_0; 
    
    int big_iter = 0;
    for (int it=0; it<niter; it++) {
        
        Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(dat.rows(), 0, dat.rows());
        std::random_shuffle(indices.data(), indices.data() + dat.rows());
        dat_temp = indices.asPermutation() * dat;
        nu = eta_chol * rnorm_vec(p);
        int count_lower = 0;
        int count_upper = batch_size;
        for (int b=0; b<nbatches; b++){
            int batch_ind = 0;
            for (int ind_temp=count_lower; ind_temp<count_upper; ind_temp++){
                dat_batch.row(batch_ind) = dat_temp.row(ind_temp);
                batch_ind += 1;
            }
            count_lower += batch_size;
            count_upper += batch_size;
            if (gradU_choice == "fig1"){
                gradU_batch = gradU_noisyFig1(theta);
            } else if (gradU_choice == "mixture_of_normals"){
                gradU_batch = gradU_mixNormals(theta, dat_batch, n, batch_size);
            } else {
                return theta_samps;
            }
            nu = nu - eta * gradU_batch - alpha * nu + Sig_chol * rnorm_vec(p);
            theta = theta + nu;
            theta_samps.col(big_iter) = theta;
            big_iter += 1;
        }
    }

    return theta_samps;
}
    
PYBIND11_MODULE(project, m) {
    m.doc() = "auto-compiled c++ extension";
    m.def("sghmc", &sghmc);
}