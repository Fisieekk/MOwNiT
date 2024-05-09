def integrator_f3(f,exact):
    errors_gk = []
    evaluations_gk = []
    errors_traps = []
    evaluations_traps = []
    for i in range(15):
        result_gk, error_estimate_gk, num_func_evals_gk = quad_vec(f, 0, 1,quadrature='gk15', epsrel=(1/10)**i,full_output=True)
        errors_gk.append(f3_error(result_gk))
        evaluations_gk.append(num_func_evals_gk.neval)
        result_trapz, error_estimate_trapz, num_func_evals_trapz = quad_vec(f, 0, 1, quadrature='trapezoid', epsabs=(1/10)**i,full_output=True)
        errors_traps.append(f3_error(result_trapz))
        evaluations_traps.append(num_func_evals_trapz.neval)
    return errors_gk, evaluations_gk, errors_traps, evaluations_traps




errors_gk2, evaluations_gk2,errors_trap2,evaluations_trap2 = integrator_f3(f3,6)
# Plotting
plt.figure(figsize=(10, 6))
plt.loglog(evaluations_gk2, errors_gk2,label='Gauss-Kronrod')
plt.loglog(evaluations_trap2, errors_trap2,label='Trapezoidal')
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Error')
plt.title('Error vs Number of Function Evaluations')
plt.grid(True)
plt.show()
