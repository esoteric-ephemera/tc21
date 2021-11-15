module ra_local_ff

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp

  real(dp), parameter :: alpha = (4._dp/(9._dp*pi))**(1._dp/3._dp)

contains

  function g0_unp_pw92_pade(rs)

    implicit none
    real(dp), intent(in) :: rs
    real(dp) :: g0_unp_pw92_pade
    real(dp), parameter :: alpha = 0.193_dp, beta = 0.525_dp

    g0_unp_pw92_pade = 0.5_dp*(1._dp + 2*alpha*rs)/(1._dp + rs*(beta &
    &     + rs*alpha*beta))**2

    !see Eq. 29 of
    !  J. P. Perdew and Y. Wang,
    !    Phys. Rev. B 46, 12947 (1992),
    !    https://doi.org/10.1103/PhysRevB.46.12947
    !    and erratum Phys. Rev. B 56, 7018 (1997)
    !    https://doi.org/10.1103/PhysRevB.56.7018
    !    NB the erratum only corrects the value of the a3
    !    parameter in gc(rs, zeta, kf R)

  end function g0_unp_pw92_pade


  subroutine ec_pw92_for_ra(z,rs,ec, decdrs, ddecdrs2, ddecdz2)

    !    Richardson-Ashcroft LFF needs some special derivatives of epsc, and
    !    moreover, needs them in Rydbergs, instead of Hartree.
    !    This routine gives those special derivatives in Rydberg

    !    J.P. Perdew and Y. Wang,
    !    ``Accurate and simple analytic representation of the electron-gas
    !          correlation energy'',
    !    Phys. Rev. B 45, 13244 (1992).
    !    https://doi.org/10.1103/PhysRevB.45.13244

    implicit none
    real(dp), intent(in) :: z, rs
    real(dp), intent(out) :: ec, decdrs, ddecdrs2, ddecdz2

    real(dp), parameter :: unp_pars(6) = (/ 0.031091,0.21370,7.5957,3.5876,1.6382,0.49294 /)
    real(dp), parameter :: pol_pars(6) = (/ 0.015545,0.20548,14.1189,6.1977,3.3662,0.62517 /)
    real(dp), parameter :: alp_pars(6) = (/ 0.016887,0.11125,10.357,3.6231,0.88026,0.49671 /)
    real(dp), parameter :: fz_den = 2._dp*(2._dp**(1._dp/3._dp)-1._dp)
    real(dp), parameter :: fdd0 = 8._dp/9._dp/fz_den

    real(dp) :: opz,omz, dxz, d_dxz_dz, d2_dxz_dz2, fz, d_fz_dz, d2_fz_dz2
    real(dp) :: ec0,d_ec0_drs,d_ec0_drs2, ec1,d_ec1_drs,d_ec1_drs2
    real(dp) :: ac,d_ac_drs,d_ac_drs2, z4, fzz4

    opz = min(2._dp,max(0._dp,1._dp+z))
    omz = min(2._dp,max(0._dp,1._dp-z))
    dxz = (opz**(4._dp/3._dp) + omz**(4._dp/3._dp))/2._dp
    d_dxz_dz = 2._dp/3._dp*(opz**(1._dp/3._dp) - omz**(1._dp/3._dp))
    d2_dxz_dz2 = 2._dp/9._dp*(opz**(-2._dp/3._dp) + omz**(-2._dp/3._dp))

    fz = 2*(dxz - 1._dp)/fz_den
    d_fz_dz = 2*d_dxz_dz/fz_den
    d2_fz_dz2 = 2*d2_dxz_dz2/fz_den

    call pw92_g_w_2nd_deriv(rs,unp_pars,ec0,d_ec0_drs,d_ec0_drs2)
    call pw92_g_w_2nd_deriv(rs,pol_pars,ec1,d_ec1_drs,d_ec1_drs2)
    call pw92_g_w_2nd_deriv(rs,alp_pars,ac,d_ac_drs,d_ac_drs2)

    z4 = z**4
    fzz4 = fz*z4

    ec = ec0 - ac/fdd0*(fz - fzz4) + (ec1 - ec0)*fzz4

    decdrs = d_ec0_drs*(1._dp - fzz4) + d_ec1_drs*fzz4 - d_ac_drs/fdd0*(fz - fzz4)
  !  decdz = -ac*d_fz_dz/fdd0 + (4*fz*z**3 + d_fz_dz*z4)*(ac/fdd0 + ec1 - ec0)

    ddecdrs2 = d_ec0_drs2*(1._dp - fzz4) + d_ec1_drs2*fzz4 - d_ac_drs2/fdd0*(fz - fzz4)
    ddecdz2 = -ac*d2_fz_dz2/fdd0 + (12*fz*z**2 + 8*d_fz_dz*z**3 + d2_fz_dz2*z4) &
    &    *(ac/fdd0 + ec1 - ec0)

    ! hartree to rydberg
    ec = 2*ec
    decdrs = 2*decdrs
    ddecdrs2 = 2*ddecdrs2
    ddecdz2 = 2*ddecdz2

  end subroutine ec_pw92_for_ra



  subroutine pw92_g_w_2nd_deriv(rs,v,g, dg, ddg)

    implicit none

    real(dp), intent(in) :: rs, v(6)
    real(dp), intent(out) :: g, dg, ddg
    real(dp) :: rsh, q0, dq0, q1, dq1, ddq1, q2, dq2, ddq2

    rsh = rs**(0.5_dp)

    q0 = -2*v(1)*(1._dp + v(2)*rs)
    dq0 = -2*v(1)*v(2)

    q1 = 2*v(1)*(v(3)*rsh + v(4)*rs + v(5)*rs*rsh + v(6)*rs*rs)
    dq1 = v(1)*(v(3)/rsh + 2*v(4) + 3*v(5)*rsh + 4*v(6)*rs)
    ddq1 = v(1)*(-0.5_dp*v(3)/rsh**3 + 3._dp/2._dp*v(5)/rsh + 4*v(6))

    q2 = log(1._dp + 1._dp/q1)
    dq2 = -dq1/(q1**2 + q1)
    ddq2 = (dq1**2*(1 + 2*q1)/(q1**2 + q1) - ddq1)/(q1**2 + q1)

    g = q0*q2
    dg = dq0*q2 + q0*dq2
    ddg = 2*dq0*dq2 + q0*ddq2

  end subroutine pw92_g_w_2nd_deriv


  subroutine fxc_ra(q,w,nw,rs,fxc)

    implicit none
    integer, intent(in) :: nw
    real(dp), intent(in) :: q,w(nw),rs
    real(dp), intent(out) :: fxc(nw)

    real(dp) :: kf, gs(nw), gn(nw), z, u(nw)

    !    NB: RA LFF expects q = (wavevector in a.u.)/(2*kf)
    !        w = (frequency in a.u.)/(2*kf**2)
    !    lff_ra_symm and lff_ra_occ return G/q**2

    !    C.F. Richardson and N.W. Ashcroft,
    !        Phys. Rev. B 50, 8170 (1994),
    !    and
    !    Eq. 32 of M. Lein, E.K.U. Gross, and J.P. Perdew,
    !        Phys. Rev. B 61, 13431 (2000)

    kf = (9*pi/4._dp)**(1._dp/3._dp)/rs
    z = q/(2*kf)
    u = w/(2*kf**2)

    call lff_ra_symm(z,u,nw,rs,gs)
    call lff_ra_occ(z,u,nw,rs,gn)
    fxc = -4*pi*(gs + gn)

  end subroutine fxc_ra


  subroutine lff_ra_symm(q,w,nw,rs,g_s)

    ! NB: q = (wavevector in a.u.)/(2*kf), w = (frequency in a.u.)/(2*kf**2)

    ! There are at least three alpha's in the RA paper
    ! alpha is determined from exact constraints, and is used in the lambdas (lam_)
    ! alp is a parameter used to control the parameterization, and is used in a, b and c

    implicit none
    integer, intent(in) :: nw
    real(dp), intent(in) :: q,w(nw),rs
    real(dp), intent(out) :: g_s(nw)

    real(dp), parameter :: alp = 0.9_dp

    real(dp) :: fac, ec, d_ec_drs, d_ec_drs2, d_ec_dz2
    real(dp) :: lam_s_inf, lam_pade, lam_n_0, lam_s_0, g0, omg0, gam_s, q2, q6
    real(dp), dimension(nw) :: a_s, c_s, b_s

    fac = ( 2._dp*(9*pi/4._dp)**(1._dp/3._dp)/rs)**2 ! = (2*kF)**2

    call ec_pw92_for_ra(0._dp,rs, ec, d_ec_drs, d_ec_drs2, d_ec_dz2)

    ! Eq. 40, corrected by Lein, Gross, and Perdew
    lam_s_inf = 3._dp/5._dp - 2._dp*pi*alpha*rs/5._dp*(rs*d_ec_drs + 2*ec)

    ! Eq. 44
    lam_pade = -0.11_dp*rs/(1._dp + 0.33_dp*rs)
    ! Eq. 39, corrected by Lein, Gross, and Perdew
    lam_n_0 = lam_pade*(1._dp - 3*(2*pi/3)**(2/3)*rs*d_ec_dz2)
    ! Eq. 38
    lam_s_0 = 1._dp + pi/3._dp*alpha*rs**2*(d_ec_drs - rs*d_ec_drs2/2._dp) - lam_n_0

    g0 = g0_unp_pw92_pade(rs)
    omg0 = 1._dp - g0

    gam_s = 9._dp/16._dp*omg0*lam_s_inf + (1._dp + 3*(1._dp - 1._dp/alp))/4._dp

    ! Eq. 56
    a_s = lam_s_inf + (lam_s_0 - lam_s_inf)/(1._dp + (gam_s*w)**2)
    ! Eq. 55
    c_s = 3._dp*lam_s_inf/(4._dp*omg0) - (4._dp/3._dp - 1._dp/alp + &
    &    3._dp*lam_s_inf/(4._dp*omg0))/(1._dp + gam_s*w)
    ! Eq. 54
    b_s = a_s/( ( (3*a_s - 2*c_s*omg0)*(1._dp + w) - 8._dp/3._dp*omg0 )* &
    &   (1._dp + w)**3 )

    q2 = q**2
    q6 = q2**3
    ! Eq. 53
    g_s = (a_s + 2._dp/3._dp*b_s*omg0*q6)/(1._dp + q2*(c_s + b_s*q6))/fac

  end subroutine lff_ra_symm



  subroutine lff_ra_occ(q,w,nw,rs,g_n)

    !    NB: q = (wavevector in a.u.)/(2*kf), w = (frequency in a.u.)/(2*kf**2)

    implicit none
    integer, intent(in) :: nw
    real(dp), intent(in) :: q,w(nw),rs
    real(dp), intent(out) :: g_n(nw)

    real(dp), parameter :: gam_n = 0.68_dp

    real(dp) :: ec, d_ec_drs, d_ec_drs2, d_ec_dz2, lam_pade, lam_n_0, lam_n_inf
    real(dp) :: fac, q2, q4
    real(dp), dimension(nw) :: gnw, gnw2, opgnw, a_n, c_n, b_n, bt

    fac = ( 2._dp*(9*pi/4._dp)**(1._dp/3._dp)/rs)**2 ! = (2*kF)**2

    gnw = gam_n*w
    gnw2 = gnw*gnw
    opgnw = 1 + gnw

    call ec_pw92_for_ra(0._dp,rs, ec, d_ec_drs, d_ec_drs2, d_ec_dz2)

    ! Eq. 44
    lam_pade = -0.11_dp*rs/(1._dp + 0.33_dp*rs)
    ! Eq. 39, corrected by Lein, Gross, and Perdew
    lam_n_0 = lam_pade*(1._dp - 3*(2*pi/3._dp)**(2/3)*rs*d_ec_dz2)
    ! Eq. 43
    lam_n_inf = 3*pi*alpha*rs*(ec + rs*d_ec_drs)

    ! Eq. 65. Note that there is a "gamma" instead of "gamma_n" in the printed version of a_n
    ! assuming this just means gamma_n

    a_n = lam_n_inf + (lam_n_0 - lam_n_inf)/(1._dp + gnw2)

    ! Eq. 64
    ! in this equation, "gam_n(w)" is printed twice. I'm assuming this just means
    ! gam_n, since this is constant. That seems to give OK agreement with their figure

    c_n = 3*gam_n/(1.18_dp*opgnw) - ( (lam_n_0 + lam_n_inf/3._dp)/(lam_n_0 + &
    &   2*lam_n_inf/3._dp) + 3*gam_n/(1.18_dp*opgnw))/(1._dp + gnw2)

    ! Eq. 63
    bt = a_n + lam_n_inf*(1._dp + 2._dp/3._dp*c_n*opgnw)
    b_n = -3._dp/(2*lam_n_inf*opgnw**2)*( bt + (bt**2 + &
    &     4._dp/3._dp*a_n*lam_n_inf)**(0.5_dp) )

    q2 = q**2
    q4 = q2*q2
    ! Eq. 62
    g_n = (a_n - lam_n_inf/3 * b_n*q4)/(1 + q2*(c_n + q2*b_n))/fac

  end subroutine lff_ra_occ


end module ra_local_ff
