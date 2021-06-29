
subroutine tc21_dynamic(q,freq,nw,rs,fxc)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp
  real(dp), parameter :: ca = 4.01_dp, cb = 1.21_dp, cc = 0.11_dp, cd = 1.07_dp

  integer, intent(in) :: nw
  real(dp), intent(in) :: q,freq(nw),rs
  complex(dp), intent(out) :: fxc

  real(dp) :: kf,fxcq,f0,kscr,tmp,fscl,qdim2
  complex(dp) :: fxcw

  call mcp07_static(q,rs,'PW92',fxcq,f0,tmp)

  kf = (9*pi/4._dp)**(1._dp/3._dp)/rs
  kscr = ca*kf/( 1._dp + cb*kf**(0.5_dp) )
  qdim2 = (q/kscr)**2
  fscl = cc*rs**2 + (1._dp - cc*rs**2)*exp(-cd*qdim2)

  call gki_dynamic_real_freq(rs,fscl*freq,nw,'PW92',.true.,fxcw)

  fxc = (1._dp + exp(-qdim2)*(fxcw/f0 - 1._dp))*fxcq

end subroutine tc21_dynamic


subroutine fxc_tc21_ifreq(q,w,nw,rs,digrid,diwg,ng,ifxc)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp
  real(dp), parameter :: ca = 4.01_dp, cb = 1.21_dp, cc = 0.11_dp, cd = 1.07_dp

  integer, intent(in) :: ng,nw
  real(dp), intent(in) :: q,rs,w(nw)
  real(dp), dimension(ng), intent(in) :: digrid,diwg
  real(dp), intent(out) :: ifxc(nw)

  integer :: iw
  real(dp) :: finf,tmp,ifxcw(nw)
  real(dp) :: kf,fxcq,f0,kscr,fscl,qdim2
  real(dp), dimension(ng) :: integrand,denom
  complex(dp), dimension(ng) :: fxc_tmp

  call mcp07_static(q,rs,'PW92',fxcq,f0,tmp)

  kf = (9*pi/4._dp)**(1._dp/3._dp)/rs
  kscr = ca*kf/( 1._dp + cb*kf**(0.5_dp) )
  qdim2 = (q/kscr)**2
  fscl = cc*rs**2 + (1._dp - cc*rs**2)*exp(-cd*qdim2)

  call high_freq(rs,'PW92',finf,tmp)

  call gki_dynamic_real_freq(rs,fscl*digrid,ng,'PW92',.true.,fxc_tmp)

  do iw = 1,nw
    denom = digrid**2 + w(iw)**2
    integrand = w(iw)*(real(fxc_tmp) - finf) + digrid*aimag(fxc_tmp)

    ifxcw(iw) = dot_product(diwg, integrand/denom)/(2*pi) + finf
  end do

  ifxc = (1._dp + exp(-qdim2)*(ifxcw/f0 - 1._dp))*fxcq

end subroutine fxc_tc21_ifreq
