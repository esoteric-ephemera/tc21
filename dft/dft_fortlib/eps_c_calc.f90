
subroutine get_eps_c(ninf,nlam,rs,ec_rpa,ec_alda,&
  &    ec_mcp07,ec_tc21,ec_cp07,ec_ra)

  use ra_local_ff, only : fxc_ra
  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  integer, intent(in) :: ninf, nlam
  real(dp), intent(in) :: rs
  real(dp),intent(out) :: ec_rpa, ec_alda, ec_mcp07,ec_tc21, ec_cp07,ec_ra

  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp

  integer :: iq,ilam,iw,ndi,nq
  real(dp) :: aq,alam, rscl,vcscl,fxc,fxcq,fxch,cwg,f0,qv0,akn
  real(dp), dimension(2*nlam) :: fxcv, fxchv,wscl
  complex(dp), dimension(2*nlam) :: fxcvc
  real(dp), dimension(4*ninf) :: digr,diwg
  real(dp), dimension(2*ninf) :: igr,iwg
  real(dp), dimension(2*nlam) :: qgr,qwg,wgr,wwg,vc,intgd
  !complex(dp), dimension(2*nlam) :: intgdc
  real(dp), dimension(nlam) :: lgr,lwg
  real(dp), dimension(2*nlam,2*nlam) :: chi0
  real(dp), dimension(nlam,2*nlam) :: dlda_iw

  ndi=4*ninf ; nq = 2*nlam

  ec_rpa = 0._dp ; ec_alda = 0._dp ; ec_mcp07 = 0._dp ; ec_tc21 = 0._dp

  call grid_gen(ninf,nlam,rs,digr,diwg,&
  &       qgr,qwg,wgr,wwg,lgr,lwg)
  igr = digr(1:2*ninf)
  iwg = diwg(1:2*ninf)

  ! unscaled Hartree kernel; vc*lambda is scaled kernel
  vc = 4*pi/qgr**2

  ! Chi_KS on imaginary frequency axis
  do iq = 1,nq
    call chi_ks_ifreq(qgr(iq),wgr,rs,nq,chi0(iq,:))
  end do


  do ilam = 1,nlam
    ! scaled frequency
    wscl = wgr/lgr(ilam)**2
    ! scaled density
    rscl = rs*lgr(ilam)
    ! Get GKI dynamic LDA on imaginary frequency axis using Cauchy integral
    call fxc_gki_ifreq(wscl,rscl,nq,digr,diwg,ndi,dlda_iw(ilam,:))
  end do

  do iq = 1,nq

    do ilam = 1,nlam

      ! integration weight at each q, lambda pair
      cwg = qwg(iq)*lwg(ilam)*4*rs**3/(3*pi)
      ! scaling q_lambda = q/lambda, rs_lambda = rs*lambda
      alam = lgr(ilam)
      aq = qgr(iq)/alam
      rscl = rs*alam
      ! vc_lambda(q) = lambda*vc(q)
      vcscl = vc(iq)*alam

      !=========================================================================
      ! RPA

      fxch = vcscl
      intgd = chi0(iq,:)**2*fxch/(1._dp - chi0(iq,:)*fxch)

      ec_rpa = ec_rpa - dot_product(wwg,intgd)*cwg

      !=========================================================================
      ! ALDA

      call alda(rscl,'PZ81',fxc)
      fxch = vcscl + fxc/alam
      intgd = chi0(iq,:)**2*fxch/(1._dp - chi0(iq,:)*fxch)

      ec_alda = ec_alda - dot_product(wwg,intgd)*cwg

      !=========================================================================
      ! MCP07, dynamic

      call mcp07_static(aq,rscl,'PZ81',fxcq,f0,akn)
      fxcv = (1._dp + exp(-akn*aq**2)*(dlda_iw(ilam,:)/f0 - 1._dp))*fxcq
      fxchv = vcscl + fxcv/alam
      intgd = chi0(iq,:)**2*fxchv(:)/(1._dp - chi0(iq,:)*fxchv(:))
      ec_mcp07 = ec_mcp07 - dot_product(wwg,intgd)*cwg

      !=========================================================================
      ! TC21

      wscl = wgr/alam**2
      call tc21_dynamic(aq,wscl,nq,'IMAG',rscl,fxcvc)
      fxchv = vcscl + real(fxcvc)/alam
      intgd = chi0(iq,:)**2*fxchv(:)/(1._dp - chi0(iq,:)*fxchv(:))
      ec_tc21 = ec_tc21 - dot_product(wwg,intgd)*cwg

      !=========================================================================
      ! CP07

      call fxc_cp07_ifreq(rscl,aq,wscl,nq,fxcv)
      fxchv = vcscl + fxcv/alam
      intgd = chi0(iq,:)**2*fxchv(:)/(1._dp - chi0(iq,:)*fxchv(:))
      ec_cp07 = ec_cp07 - dot_product(wwg,intgd)*cwg

      !=========================================================================
      ! RA

      call  fxc_ra(aq,wscl,nq,rscl,fxcv)
      fxchv = vcscl + fxcv/alam
      intgd = chi0(iq,:)**2*fxchv(:)/(1._dp - chi0(iq,:)*fxchv(:))
      ec_ra = ec_ra - dot_product(wwg,intgd)*cwg

    end do
  end do

end subroutine get_eps_c



subroutine grid_gen(cpts,vpts,rs,digrid,diwg,&
&       qgr,qwg,wgr,wwg,lgrid,lwg)

  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  real(dp), parameter :: pi = 3.14159265358979323846264338327950288419_dp

  integer, intent(in) :: cpts,vpts
  real(dp), intent(in) :: rs
  real(dp),dimension(4*cpts), intent(out) :: digrid, diwg
  real(dp),dimension(vpts), intent(out) :: lgrid,lwg
  real(dp),dimension(2*vpts), intent(out) :: wgr,wwg,qgr,qwg

  real(dp) :: cut_pt,wp0,kf
  real(dp), dimension(cpts) :: sgrid,swg

  call gauss_legendre(cpts,sgrid,swg)

  sgrid = 0.5_dp*(sgrid + 1._dp)
  swg = 0.5_dp*swg

  wp0 = (3._dp/rs**3)**(0.5_dp)
  cut_pt = 100._dp*wp0

  digrid(1:cpts) = cut_pt*sgrid
  diwg(1:cpts) = cut_pt*swg
  digrid(cpts+1:2*cpts) = cut_pt/sgrid
  diwg(cpts+1:2*cpts) = swg*digrid(cpts+1:2*cpts)**2/cut_pt
  digrid(2*cpts+1:4*cpts) = -digrid(1:2*cpts)
  diwg(2*cpts+1:4*cpts) = diwg(1:2*cpts)


  call gauss_legendre(vpts,lgrid,lwg)

  lgrid = 0.5_dp*(lgrid + 1._dp)
  lwg = 0.5_dp*lwg

  wp0 = (3._dp/rs**3)**(0.5_dp)
  cut_pt = 100._dp*wp0

  wgr(1:vpts) = cut_pt*lgrid
  wwg(1:vpts) = cut_pt*lwg
  wgr(vpts+1:2*vpts) = cut_pt + wgr(1:vpts)
  wwg(vpts+1:2*vpts) = wwg(1:vpts)

  kf = (9*pi/4._dp)**(1._dp/3._dp)/rs
  cut_pt = 100._dp*kf

  qgr(1:vpts) = cut_pt*lgrid
  qwg(1:vpts) = cut_pt*lwg
  qgr(vpts+1:2*vpts) = cut_pt + qgr(1:vpts)!cut_pt/lgrid
  qwg(vpts+1:2*vpts) = qwg(1:vpts)!lwg*qgr(vpts+1:2*vpts)**2/cut_pt


end subroutine grid_gen
