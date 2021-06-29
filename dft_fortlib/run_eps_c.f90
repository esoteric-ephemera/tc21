program jellium_ec

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp),parameter :: pi = 3.14159265358979323846264338327950288419_dp

  integer, parameter :: ninf=200, nlam=100
!  real(dp), parameter :: rs_min=1._dp,rs_max = 10._dp,drs = 0.1_dp

  integer, parameter :: nrs = 129!ceiling((rs_max-rs_min)/drs)

  real(dp), dimension(nrs) :: rsl, ec_rpa, ec_alda,ec_mcp07,ec_tc21
  real(dp) :: ecpw92
  integer :: irs

  character(len=1000) :: str

  do irs =1,9
    rsl(irs) = 1._dp*irs/10._dp
  end do
  do irs = 1,120
    rsl(9+irs) = 1._dp*irs
  end do

  !$OMP PARALLEL DO
  do irs = 1,nrs
    !rsl(irs) = rs_min + drs*irs
    call get_eps_c(ninf,nlam,rsl(irs),ec_rpa(irs),ec_alda(irs),ec_mcp07(irs)&
   &    ,ec_tc21(irs))
    print*,ec_rpa(irs),ec_alda(irs)
    print*,ec_mcp07(irs),ec_tc21(irs)
  end do
  !$OMP END PARALLEL DO
  !stop

  open(unit=2,file='jell_eps_c.csv')
  write(2,'(a)') 'rs, PW92, RPA, ALDA, MCP07, TC21'
  ! unfortunately the second do loop is needed because the multithreading removes
  ! the ordering of the rs values
  do irs = 1,nrs
    call ec_pw92_unpol(rsl(irs),ecpw92)
    write(str,*) rsl(irs),',',ecpw92,',',ec_rpa(irs),',',ec_alda(irs),',',ec_mcp07(irs),',',&
    &    ec_tc21(irs)
    write(2,'(a)') trim(adjustl(str))
  end do

  close(2)

end program jellium_ec


subroutine ec_pw92_unpol(rs,epsc)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  real(dp), intent(in) :: rs
  real(dp), intent(out) :: epsc

  call pw92_g(rs,0.031091_dp,0.21370_dp,7.5957_dp,3.5876_dp,1.6382_dp,&
 &     0.49294_dp,epsc)

end subroutine ec_pw92_unpol


subroutine pw92_g(rs,apar,alpha,b1,b2,b3,b4,g)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  real(dp), intent(in) :: rs,apar,alpha,b1,b2,b3,b4
  real(dp), intent(out) :: g

  real(dp) :: q0,q1,q2,rsh

  rsh = rs**(0.5_dp)

  q0 = -2._dp*apar*(1._dp + alpha*rs)
  q1 = 2._dp*apar*(b1*rsh + b2*rs + b3*rs*rsh + b4*rs**2)
  q2 = log(1._dp + 1._dp/q1)
  g = q0*q2

end subroutine pw92_g
