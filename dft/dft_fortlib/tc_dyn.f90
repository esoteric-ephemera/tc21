
subroutine tc21_dynamic(q,freq,nw,freqax,rs,fxc)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp
  real(dp), parameter :: ca = 3.846991_dp, cb = 0.471351_dp, cc = 4.346063_dp
  real(dp), parameter :: cd = 0.881313_dp

  integer, intent(in) :: nw
  real(dp), intent(in) :: q,freq(nw),rs
  character(len=4), intent(in) :: freqax
  complex(dp), intent(out) :: fxc(nw)

  real(dp) :: kf,fxcq,f0,kscr,tmp,pscl,fscl,qdim2
  complex(dp) :: fxcw(nw)

  call mcp07_static(q,rs,'PW92',fxcq,f0,tmp)

  kf = (9*pi/4._dp)**(1._dp/3._dp)/rs
  kscr = kf*(ca + cb*kf**(1.5_dp))/( 1._dp + kf**2 )
  qdim2 = (q/kscr)**2
  fscl = (rs/cc)**2
  pscl = fscl + (1._dp - fscl)*exp(-cd*qdim2)

  call gki_dynamic(rs,pscl*freq,nw,freqax,fxcw)

  fxc = (1._dp + exp(-qdim2)*(fxcw/f0 - 1._dp))*fxcq

end subroutine tc21_dynamic


subroutine gki_dynamic(rs,freq,nw,axis,fxc)

  ! TC21 parameterization of GKI kernel
  ! for purely real, or purely imaginary frequencies
  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp
  real(dp), parameter :: gam = 1.311028777146059809410871821455657482147216796875_dp
  real(dp), parameter :: cc = 4.81710873550434914847073741839267313480377197265625_dp

  integer, intent(in) :: nw
  real(dp), intent(in) :: rs
  real(dp), dimension(nw), intent(in) :: freq
  character(len=4), intent(in) :: axis
  complex(dp), dimension(nw), intent(out) :: fxc

  real(dp), parameter, dimension(4) :: cp = (/0.174724_dp, 3.224459_dp, 2.221196_dp, &
   &    1.891998_dp /)
  real(dp), parameter, dimension(5) :: kp = (/1.219946_dp, 0.973063_dp, 0.42106_dp, &
   &    1.301184_dp, 1.007578_dp /)

  real(dp) :: bn,finf
  real(dp), dimension(nw) :: u,gx,hx,jx

  fxc = 0._dp
  call high_freq(rs,'PW92',finf,bn)
  u = bn**(0.5_dp)*freq

  if (axis == 'REAL') then

    gx = u/(1._dp + u**2)**(5._dp/4._dp)

    hx = 1._dp/gam*(1._dp - cp(1)*u**2)/(1 + cp(2)*u**2 + cp(3)*u**4 &
   &     + cp(4)*u**6 + (cp(1)/gam)**(16._dp/7._dp)*u**8)**(7._dp/16._dp)

    fxc = finf - cc*bn**(3._dp/4._dp)*(hx + gx*cmplx(0._dp,1._dp))

  else if (axis == 'IMAG') then

    jx = 1._dp/gam*(1._dp - kp(1)*u + kp(2)*u**2)/(1._dp + kp(3)*u**2 + kp(4)*u**4 &
   &     + kp(5)*u**6 + (kp(1)/gam)**(16._dp/7._dp)*u**8)**(7._dp/16._dp)
    fxc = finf - cc*bn**(3._dp/4._dp)*jx! + 0._dp*cmplx(0._dp,1._dp)

  else

    print*,'Unknown frequnecy axis ',axis,' in revised GKI!'
    stop

  end if

end subroutine gki_dynamic
