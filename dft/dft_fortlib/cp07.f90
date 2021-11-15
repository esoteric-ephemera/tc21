
subroutine fxc_cp07_ifreq(rs,q,u,nu,fxc)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp
  real(dp), parameter :: qeps = 1.e-6, gam1 = -0.114548231_dp, gam2 = 0.614523371_dp
  real(dp), parameter :: alp = -0.0255184916_dp, bet = 0.691590707_dp

  !    NB: u = Im(omega), where Re(omega) = 0
  !    L.A. Constantin and J.M. Pitarke,
  !        Phys. Rev. B 75, 245127 (2007).
  !        doi: 10.1103/PhysRevB.75.245127

  integer, intent(in) :: nu
  real(dp), intent(in) :: q,u(nu),rs
  real(dp), intent(out) :: fxc(nu)

  real(dp) :: ec,d_ec_drs,kf,rsh,ars,brs,cn, crs,fxc_alda, fxc_inf, n, q2
  real(dp),dimension(nu) :: ku, kuq2

    call lda_derivs(rs,'PW92',ec,d_ec_drs)
    kf = (9*pi/4._dp)**(1._dp/3._dp)/rs

    crs = -pi/(2*kf)*(ec + rs*d_ec_drs)

    ! brs according to the parametrization of Eq. (7) of
    ! Massimiliano Corradini, Rodolfo Del Sole, Giovanni Onida, and Maurizia Palummo
    ! Phys. Rev. B 57, 14569 (1998)
    ! doi: 10.1103/PhysRevB.57.14569
    rsh = rs**(0.5_dp)
    brs = (1._dp + 2.15_dp*rsh + 0.435_dp*rsh**3)/(3._dp + 1.57_dp*rsh &
    &     + 0.409_dp*rsh**3)

    ! Eq. A4
    ars = exp(10.5_dp/(1._dp + rs)**(13._dp/2._dp)) + 0.5_dp

    ! Eq. 8
    n = 3._dp/(4*pi*rs**3)
    fxc_inf = gam1/n**gam2

    ! Eq. 6
    fxc_alda = 4*pi*alp/n**bet

    ! Eq. 10
    cn = fxc_inf/fxc_alda
    ! Eq. A3
    ku = -fxc_alda/(4*pi*brs)*(1 + u*(ars + u*crs))/(1 + u**2)

    q2 = q**2
    kuq2 = ku*q2
    ! Eq. 12

    if (q < qeps) then
      fxc = 4*pi*brs*ku*(-1._dp + kuq2*(0.5_dp - kuq2/6._dp) ) &
      &  - 4*pi*crs*q2*(1._dp - q2)/kf**2
    else
      fxc = 4*pi*brs/q2*(exp(-kuq2)-1._dp) - 4*pi/kf**2 * crs/(1._dp + 1._dp/q2)
    end if

end subroutine fxc_cp07_ifreq
