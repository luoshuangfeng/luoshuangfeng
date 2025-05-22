/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2022 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Authors: Manaswinee Bezbaruah, Matthias Maier, Texas A&M University, 2021.
 */

//pmb

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function_parser.h>


#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>


#include <fstream>
#include <iostream>
#include <memory>
#include<complex>


namespace Step81
{
  using namespace dealii;
  using namespace std::complex_literals;






  template <int dim>
  class Parameters : public ParameterAcceptor
  {
  public:
    Parameters();

    using rank0_type = std::complex<double>;

    using rank1_type = Tensor<1, dim, std::complex<double>>;

    using rank2_type = Tensor<2, dim, rank0_type>;

    using curl_type = Tensor<1, dim == 2 ? 1 : dim, rank0_type>;
    double tem;
  public:

    std::complex<double> curl_incident_field(const Point<dim> &point, double t) const;
    rank2_type epsilon(const Point<dim> &x, types::material_id material);

    rank2_type epsilon_r(const Point<dim> &x, types::material_id material);

    std::complex<double> mu_inv(const Point<dim>  &x,
                                types::material_id material);

    std::complex<double> mu_inv_r(const Point<dim>  &x,
                                    types::material_id material);

    rank2_type sigma(const Point<dim>  &x,
                     types::material_id left,
                     types::material_id right);

    rank1_type J_a(const Point<dim> &point, types::material_id id);

    Tensor<1, dim> get_boundary_normal(const Point<dim>& position)  ;

    Tensor<1,dim> get_edge_direction(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        unsigned int dof_index,
        const FiniteElement<dim> &fe);

    bool in_tfsf_boundary(const Point<dim>& position) const;

    bool in_tfsf_boundary_lr(const Point<dim>& position) const;

    bool in_tfsf_boundary_td(const Point<dim>& position) const;
    // 判断点是否在tot区域内
    bool is_in_hard_source_region(const Point<dim> &point) const
    {
      const double x_min = -1, x_max = 1;
      const double y_min = -1, y_max = 1;
      return (point[0] >= x_min && point[0] <= x_max &&
              point[1] >= y_min && point[1] <= y_max);
    }
    //判断点是否在material区域内
    bool in_material_region(const Point<dim> &point) const{
    	const auto radius = point.norm();

    	return (radius<=0.48&&radius>=0.24);
    }

    // 判断点是否位于总场/散射场边界区域
    bool is_on_tfsf_boundary(const Point<dim>& point) const
    {
        // 定义总场区范围（示例：中心区域）
        const double tf_x_min = -1.0, tf_x_max = 1.0;
        const double tf_y_min = -1.0, tf_y_max = 1.0;

        // 定义缓冲层厚度（PML或过渡区）
        const double tolerance = 1e-2;

        // 判断点是否在总场区边界附近
        bool near_x_boundary =
           ( ((point[0] >= tf_x_min - tolerance && point[0] <= tf_x_min) ||
            (point[0] >= tf_x_max && point[0] <= tf_x_max + tolerance))&&(point[1]<=tf_y_max&&point[1]>=tf_y_min));

        bool near_y_boundary =
           ( (point[1] >= tf_y_min - tolerance && point[1] <= tf_y_min) ||
            (point[1] >= tf_y_max && point[1] <= tf_y_max + tolerance)&&(point[0]<=tf_x_max&&point[0]>=tf_x_min));

        bool val = //(!is_in_hard_source_region(point)) &&
        		(near_x_boundary||near_y_boundary) ;
//           if(val)
//           {
//        	   double temp=1;
//
//        std::cout << "temp = " << temp<< std::endl;
// }
        // 总场边界定义为：在总场区外但紧邻其边界的区域
        return (near_x_boundary||near_y_boundary);
    }
    bool is_in_inc_source_region(const Point<dim> &point) const
    {
      const double x_min = -1.0, x_max = 1.0;
      const double y_min = -1.0, y_max = 1.0;
      return (point[0] >= x_min && point[0] <= x_max &&
              point[1] >= y_min && point[1] <= y_max);
    }

    // 电场值（平面波）
      Tensor<1, dim, std::complex<double>> hard_source_E(const Point<dim> &point, double t ) const
      {
        const double k = 14.2; // 波数
        const double phase = k * (point * plane_wave_direction)- 2 * M_PI * plane_wave_frequency * t;
        return plane_wave_amplitude * std::exp(std::complex<double>(0, phase)) * plane_wave_polarization;
      }
  private:
    rank2_type           epsilon_1;
    rank2_type           epsilon_2;
    std::complex<double> mu_inv_1;
    std::complex<double> mu_inv_2;
    rank2_type           sigma_tensor;
    double R2 = 0.48;
    double R1 = 0.24;
    double c = 0.08 ;

    double plane_wave_amplitude;   // 平面波振幅
      double plane_wave_frequency;   // 平面波频率
      Tensor<1, dim> plane_wave_direction;   // 平面波传播方向
      Tensor<1, dim> plane_wave_polarization; // 平面波极化方向

      // 添加点源参数
      Point<dim> point_source_position;
      double point_source_amplitude;
      double point_source_frequency;
  };


  template <int dim>
  Parameters<dim>::Parameters()
    : ParameterAcceptor("Parameters")
  {
    epsilon_1[0][0] = 1;
    epsilon_1[1][1] = 1;
    add_parameter("material 1 epsilon",
                  epsilon_1,
                  "relative permittivity of material 1");

    epsilon_2[0][0] = 1;
    epsilon_2[1][1] = 1;
    add_parameter("material 2 epsilon",
                  epsilon_2,
                  "relative permittivity of material 2");

    mu_inv_1 = 1;
    add_parameter("material 1 mu_inv",
                  mu_inv_1,
                  "inverse of relative permeability of material 1");

    mu_inv_2 = 1;
    add_parameter("material 2 mu_inv",
                  mu_inv_2,
                  "inverse of relative permeability of material 2");

    sigma_tensor[0][0] = std::complex<double>(0, 0);
    sigma_tensor[1][1] = std::complex<double>(0, 0);
    add_parameter("sigma",
                  sigma_tensor,
                  "surface conductivity between material 1 and material 2");

   /* dipole_radius = 4;
    add_parameter("dipole radius", dipole_radius, "radius of the dipole");

    dipole_position = Point<dim>(0., 0.);
    add_parameter("dipole position", dipole_position, "position of the dipole");

    dipole_orientation = Tensor<1, dim>{{0., 1.}};
    add_parameter("dipole orientation",
                  dipole_orientation,
                  "orientation of the dipole");

    dipole_strength = 1.;
    add_parameter("dipole strength", dipole_strength, "strength of the dipole");*/

   plane_wave_amplitude = 1.0   ;    // 平面波振幅
   add_parameter("plane wave amplitude", plane_wave_amplitude);

   plane_wave_frequency = 0.6777777e9    ;    // 平面波频率（Hz）
   add_parameter("plane wave frequency", plane_wave_frequency);

   plane_wave_direction = Tensor<1, dim>{{1, 0}};      //平面波传播方向（单位向量）
   add_parameter("plane wave direction", plane_wave_direction);

   plane_wave_polarization = Tensor<1, dim>{{0, 1}};  //平面波极化方向（单位向量）
   add_parameter("plane wave polarization", plane_wave_polarization);

   // 点源参数
     point_source_position = Point<dim>(0.0, 0.0); // 默认位置
     add_parameter("point source position", point_source_position, "Position of the point source");

     point_source_amplitude = 1.0;
     add_parameter("point source amplitude", point_source_amplitude, "Amplitude of the point source");

     point_source_frequency = 1e9; // 默认频率
     add_parameter("point source frequency", point_source_frequency, "Frequency of the point source");
  }

  template <int dim>
  typename Parameters<dim>::rank2_type
    Parameters<dim>::epsilon(const Point<dim> & /*x*/,
                             types::material_id material)
    {
      return  epsilon_1;
    }


  template <int dim>
  std::complex<double> Parameters<dim>::curl_incident_field(
      const Point<dim> &point, double t) const
  {
	    const double k = 14.2;
	    const double phase = k * (point * plane_wave_direction) - 2 * M_PI * plane_wave_frequency * t;
	    const std::complex<double> j(0, 1);


	        // 2D TE模式：返回标量 (∇×E)_z = ∂Ey/∂x - ∂Ex/∂y
	        auto E_inc = plane_wave_amplitude * plane_wave_polarization * std::exp(j * phase);
	        return j * k * (plane_wave_direction[0] * E_inc[1] - plane_wave_direction[1] * E_inc[0]);
  }

  template <int dim>
  typename Parameters<dim>::rank2_type
  Parameters<dim>::epsilon_r(const Point<dim> &point /*x*/,
                           types::material_id material)
  {
	 rank2_type epsilon_r;
	 rank2_type I;
	 I[0][0] = 1.0;
	 I[1][1] = 1.0;

	const auto  r = point.norm();
	double detA = std::pow((R2-R1)/R2 , 2)*(r/(r-R1));
    double x = point[0];
    double y = point[1];
    double theta = std::atan2(y,x);
    double k = (R2 - R1)/(R2 - c);
//   epsilon_r[0][0] = (std::pow((R2-R1)/R2,2) +
//					  (1 + 2*std::pow((R2-R1)/R2,2)*R1/(r-R1))*std::pow(std::sin(theta),2))/detA;
//
//   epsilon_r[1][0] = -((1 + 2*std::pow((R2-R1)/R2,2)*R1/(r-R1))*std::sin(theta)*std::cos(theta))/detA;
//
//   epsilon_r[0][1] = -((1 + 2*std::pow((R2-R1)/R2,2)*R1/(r-R1))*std::sin(theta)*std::cos(theta))/detA;
//
//   epsilon_r[1][1] = (std::pow((R2-R1)/R2,2) +
//					  (1 + 2*std::pow((R2-R1)/R2,2)*R1/(r-R1))*std::pow(std::cos(theta),2))/detA;

    epsilon_r[0][0] = ((r - R1 + k * c)*std::pow(std::cos(theta), 2)/r) +
    		           r * std::pow(std::sin(theta), 2)/(r - R1 + k * c);
    epsilon_r[0][1] = ((r - R1 + k * c)/r - r/(r - R1 + k * c)) * std::cos(theta) * std::sin(theta);

    epsilon_r[1][0] = ((r - R1 + k * c)/r - r/(r - R1 + k * c)) * std::cos(theta) * std::sin(theta);

    epsilon_r[1][1] = ((r - R1 + k * c)*std::pow(std::sin(theta), 2)/r) +
	           r * std::pow(std::cos(theta), 2)/(r - R1 + k * c);
    return  (in_material_region(point) ? epsilon_r : I);
  }

  template <int dim>
   std::complex<double> Parameters<dim>::mu_inv_r(const Point<dim> & point/*x*/,
                                                types::material_id material)
   {
//	  const auto  r = point.norm();
//	 double detA = std::pow((R2-R1)/R2 , 2)*(r/(r-R1));
//	  auto mu_inv = detA;
	  const auto  r = point.norm();
      double k = (R2 - R1)/(R2 - c);
      auto mu_inv = std::pow(k, 2) * r/(r - R1 + k * c);
     return (in_material_region(point)? mu_inv : 1.0);
   }

  template <int dim>
  std::complex<double> Parameters<dim>::mu_inv(const Point<dim> & /*x*/,
                                               types::material_id material)
  {
    return mu_inv_1;
  }

  template <int dim>
  typename Parameters<dim>::rank2_type
  Parameters<dim>::sigma(const Point<dim> & /*x*/,
                         types::material_id left,
                         types::material_id right)
  {
    return (left == right ? rank2_type() : sigma_tensor);
  }
  template <int dim>
  bool Parameters<dim>::in_tfsf_boundary(const Point<dim>& position) const {
      const double tol = 1e-3;
      const double tf_min = -1, tf_max = 1; // 总场区范围

      return (
          (std::abs(position[0] - tf_min) < tol && position[1] >= tf_min && position[1] <= tf_max) ||
          (std::abs(position[0] - tf_max) < tol && position[1] >= tf_min && position[1] <= tf_max) ||
          (std::abs(position[1] - tf_min) < tol && position[0] >= tf_min && position[0] <= tf_max) ||
          (std::abs(position[1] - tf_max) < tol && position[0] >= tf_min && position[0] <= tf_max)
      );
  }

  template <int dim>
    bool Parameters<dim>::in_tfsf_boundary_td(const Point<dim>& position) const {
        const double tol = 1e-3;
        const double tf_min = -1, tf_max = 1; // 总场区范围

        return (
        		(std::abs(position[1] - tf_min) < tol && position[0] >= tf_min && position[0] <= tf_max) ||
        		          (std::abs(position[1] - tf_max) < tol && position[0] >= tf_min && position[0] <= tf_max)

        );
    }

  template <int dim>
      bool Parameters<dim>::in_tfsf_boundary_lr(const Point<dim>& position) const {
          const double tol = 1e-3;
          const double tf_min = -1, tf_max = 1; // 总场区范围

          return
          		((std::abs(position[0] - tf_min) < tol && position[1] >= tf_min && position[1] <= tf_max) ||
          	          (std::abs(position[0] - tf_max) < tol && position[1] >= tf_min && position[1] <= tf_max)

          );
      }

  template <int dim>
  Tensor<1, dim> Parameters<dim>::get_boundary_normal(const Point<dim>& position)  {
      const double tol = 1e-6;
      const double tf_min = -1, tf_max = 1;

      Tensor<1, dim> normal;

      // 左边界 (法向量指向x负方向)
      if (std::abs(position[0] - tf_min) < tol && position[1] > tf_min && position[1] < tf_max) {
          normal[0] = -1.0;
          normal[1] = 0.0;
      }
      // 右边界 (法向量指向x正方向)
      else if (std::abs(position[0] - tf_max) < tol && position[1] > tf_min && position[1] < tf_max) {
          normal[0] = 1.0;
          normal[1] = 0.0;
      }
      // 下边界 (法向量指向y负方向)
      else if (std::abs(position[1] - tf_min) < tol && position[0] >= tf_min && position[0] <= tf_max) {
          normal[0] = 0.0;
          normal[1] = -1.0;
      }
      // 上边界 (法向量指向y正方向)
      else if (std::abs(position[1] - tf_max) < tol && position[0] >= tf_min && position[0] <= tf_max) {
          normal[0] = 0.0;
          normal[1] = 1.0;
      }
      // 非边界情况（默认返回零向量）
      else normal = 0 ;

      return normal;
  }

  template <int dim>
  Tensor<1,dim> Parameters<dim>::get_edge_direction(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      unsigned int dof_index,
      const FiniteElement<dim> &fe)
  {
      Assert(dim == 2, ExcNotImplemented()); // 确保仅用于2D
      // 正确获取棱元方向（适用于非原始元）
      const auto base_index = fe.system_to_base_index(dof_index);
      const unsigned int edge_index = base_index.second;

      const auto vertices = cell->line(edge_index)->vertex_indices();
     // std::cout<<vertices<<std::endl;
      Tensor<1,dim> edge_dir = cell->line(edge_index)->vertex(vertices[1]) - cell->line(edge_index)->vertex(vertices[0]);
    //  std::cout<<edge_dir<<std::endl;
      edge_dir /= edge_dir.norm();
      edge_dir /= edge_dir.norm();
      const Point<dim>& v0 = cell->line(edge_index)->vertex(vertices[0]);
      const Point<dim>& v1 = cell->line(edge_index)->vertex(vertices[1]) ;
      const double tol = 1e-8;
      if (std::abs(edge_dir[1]) < tol) {
          // 水平边：强制从左到右（+x方向）
          if (v0[0] > v1[0]) edge_dir *= -1.0;
      } else {
          // 垂直边：强制从下到上（+y方向）
          if (v0[1] > v1[1]) edge_dir *= -1.0;
      }
     // std::cout<<edge_dir<<std::endl;
      return edge_dir;
  }

  template <int dim>
  typename Parameters<dim>::rank1_type
  Parameters<dim>::J_a(const Point<dim> &point, types::material_id /*id*/)
  {
	  rank1_type J_a = rank1_type();

	    // 定义点源的高斯分布参数
	    const double width = 0.2; // 控制点源的分布宽度
	    const double x0 = point_source_position[0];
	    const double y0 = point_source_position[1];

	    // 高斯分布函数
	    const double exponent = -((point[0]-x0)*(point[0]-x0) + (point[1]-y0)*(point[1]-y0)) / (width*width);
	    const double gaussian = point_source_amplitude * std::exp(exponent);

	    // 设置极化方向（例如 y 方向）
	    J_a[1] = std::complex<double>(gaussian, 0.0); // 实部激励

	    // 若需要时谐源，例如 exp(-iωt)
	    // const std::complex<double> j(0, 1);
	    // J_a[1] = gaussian * std::exp(-j * 2 * M_PI * point_source_frequency * t);

	    return J_a;

  }



  template <int dim>
  class PerfectlyMatchedLayer : public ParameterAcceptor
  {
  public:
    static_assert(dim == 2,
                  "The perfectly matched layer is only implemented in 2d.");

    Parameters<dim> parameters;

    using rank1_type = Tensor<1, dim, std::complex<double>>;

    using rank2_type = Tensor<2, dim, std::complex<double>>;

    PerfectlyMatchedLayer();

    std::complex<double> d(const Point<dim> point);

    std::complex<double> d_bar(const Point<dim> point);

    std::complex<double> compute_s_x(const Point<dim> &point)const;

    std::complex<double> compute_s_y(const Point<dim> &point) const;

    bool is_in_pml(const Point<dim> &point) const;

    double compute_sigma_x(const Point<dim> &point) const;
    double compute_sigma_y(const Point<dim> &point) const;
    //rank2_type rotation(std::complex<double> d_1,
                //        std::complex<double> d_2,
         //               Point<dim>           point);

   //rank2_type a_matrix(const Point<dim> point);

   // rank2_type b_matrix(const Point<dim> point);

   // rank2_type c_matrix(const Point<dim> point);

  private:
    std::vector<double> inner_rectangle; // 内部矩形边界 [xmin, xmax, ymin, ymax]
    std::vector<double> outer_rectangle; // 外部矩形边界 [xmin, xmax, ymin, ymax]
    double reflection_coefficient;
    double wave_number ;
    double strength;// PML 强度
    const double cv=1.0;

  };

  template <int dim>
  PerfectlyMatchedLayer<dim>::PerfectlyMatchedLayer()
    : ParameterAcceptor("PerfectlyMatchedLayer")
  {
    // 默认内部矩形边界
    inner_rectangle = {-1.5 ,1.5 ,-1.5,1.5};
    add_parameter("inner rectangle",
                  inner_rectangle,
                  "Inner rectangle boundaries [xmin, xmax, ymin, ymax]");

    // 默认外部矩形边界
    outer_rectangle = {-2. ,2.,-2. ,2.};
    add_parameter("outer rectangle",
                  outer_rectangle,
                  "Outer rectangle boundaries [xmin, xmax, ymin, ymax]");

    // 默认 PML 强度
    strength = 0.5;
    add_parameter("strength", strength, "Strength of the PML");

    //反射率 R(0)
    reflection_coefficient=1e-6;
    add_parameter("reflection coefficient",reflection_coefficient);

    //波数
    wave_number = 14.2;
    add_parameter("wave number",wave_number);
  }



  template <int dim>

   double PerfectlyMatchedLayer<dim>::compute_sigma_x(const Point<dim> &point) const
  {

    const double x = point[0];
    const double y = point[1];

    // 计算到 PML 界面的距离
    const double rho_x = std::max(0.0, std::max(inner_rectangle[0] - x, x - inner_rectangle[1]));
   // const double rho_y = std::max(0.0, std::max(inner_rectangle[2] - y, y - inner_rectangle[3]));

    // 计算 sigma_max
//    const double sigma_max = -(5 * cv * std::log(reflection_coefficient)) / (2 * strength);
    if((y<inner_rectangle[2]&&x>inner_rectangle[0]&&x<inner_rectangle[1])||
  		  (y>inner_rectangle[3]&&x>inner_rectangle[0]&&x<inner_rectangle[1]))
       	  return 0 ;
    // 计算 sigma(x) 和 sigma(y)
    const double sigma_x = 20* std::pow(rho_x / strength, 4);
//std::cout<<"sigma_x="<<sigma_x<<std::endl;

    return sigma_x ;
  }

  template <int dim>

     double PerfectlyMatchedLayer<dim>::compute_sigma_y(const Point<dim> &point) const
    {

	  const double x = point[0];
      const double y = point[1];

      // 计算到 PML 界面的距离

      const double rho_y = std::max(0.0, std::max(inner_rectangle[2] - y, y - inner_rectangle[3]));
    //  const double rho_x = std::max(0.0, std::max(inner_rectangle[0] - x, x - inner_rectangle[1]));

      // 计算 sigma_max
     // const double sigma_max = -(5 * cv * std::log(reflection_coefficient)) / (2 * strength);
      if((x<inner_rectangle[0]&&y>inner_rectangle[2]&&y<inner_rectangle[3])||
    		  (x>inner_rectangle[1]&&y>inner_rectangle[2]&&y<inner_rectangle[3]))
    	  return 0 ;
      // 计算 sigma(x) 和 sigma(y)

      const double sigma_y = 20 * std::pow(rho_y / strength, 4);

      return sigma_y ;
    }

  template <int dim>
  typename std::complex<double>
   PerfectlyMatchedLayer<dim>::compute_s_x(const Point<dim> &point) const
  {


    const double sigma = compute_sigma_x(point);
    return (1.0 + std::complex<double>(0, sigma/13));
  }

  template <int dim>
  typename std::complex<double>
  PerfectlyMatchedLayer<dim>::compute_s_y(const Point<dim> &point) const
  {
    const double sigma = compute_sigma_y(point);
    return (1.0 + std::complex<double>(0, sigma/13 ));
  }

  template <int dim>
  bool PerfectlyMatchedLayer<dim>::is_in_pml(const Point<dim> &point) const
  {
    const double x = point[0];
    const double y = point[1];

    // 判断是否在 PML 区域内
    const bool in_pml_x = (x < inner_rectangle[0] || x >inner_rectangle[1]);
    const bool in_pml_y = (y <inner_rectangle[2] || y >inner_rectangle[3]);

    return in_pml_x || in_pml_y;
  }

  template <int dim>
  typename std::complex<double>
  PerfectlyMatchedLayer<dim>::d(const Point<dim> point)
  {
	  // 判断是否在 PML 区域内
    if (!is_in_pml(point))
	 return 1.0; // 不在 PML 区域内，无衰减
     const auto s_x = compute_s_x(point);
     const auto s_y = compute_s_y(point);
     return s_x * s_y;
  }

  template <int dim>
  typename std::complex<double>
  PerfectlyMatchedLayer<dim>::d_bar(const Point<dim> point)
  {
	  // 判断是否在 PML 区域内
	if (!is_in_pml(point))
	 return 1.0; // 不在 PML 区域内，无衰减
     const auto s_x = compute_s_x(point);
     const auto s_y = compute_s_y(point);
     return s_y / s_x;
  }

  /*template <int dim>
  typename PerfectlyMatchedLayer<dim>::rank2_type
  PerfectlyMatchedLayer<dim>::rotation(std::complex<double> d_1,
                                       std::complex<double> d_2,
                                       Point<dim>           point)
  {
    rank2_type result;
    result[0][0] = point[0] * point[0] * d_1 + point[1] * point[1] * d_2;
    result[0][1] = point[0] * point[1] * (d_1 - d_2);
    result[1][0] = point[0] * point[1] * (d_1 - d_2);
    result[1][1] = point[1] * point[1] * d_1 + point[0] * point[0] * d_2;
    return result;
  }


  template <int dim>
  typename PerfectlyMatchedLayer<dim>::rank2_type
  PerfectlyMatchedLayer<dim>::a_matrix(const Point<dim> point)
  {
    const auto d     = this->d(point);
    const auto d_bar = this->d_bar(point);
    return invert(rotation(d * d, d * d_bar, point)) *
           rotation(d * d, d * d_bar, point);
  }


  template <int dim>
  typename PerfectlyMatchedLayer<dim>::rank2_type
  PerfectlyMatchedLayer<dim>::b_matrix(const Point<dim> point)
  {
    const auto d     = this->d(point);
    const auto d_bar = this->d_bar(point);
    return invert(rotation(d, d_bar, point)) * rotation(d, d_bar, point);
  }


  template <int dim>
  typename PerfectlyMatchedLayer<dim>::rank2_type
  PerfectlyMatchedLayer<dim>::c_matrix(const Point<dim> point)
  {
    const auto d     = this->d(point);
    const auto d_bar = this->d_bar(point);
    return invert(rotation(1. / d_bar, 1. / d, point)) *
           rotation(1. / d_bar, 1. / d, point);
  }*/



  template <int dim>
  class Maxwell : public ParameterAcceptor
  {
  public:
    Maxwell();
    void run();

  private:
    /* run time parameters */
    double       scaling;
    unsigned int refinements;
    unsigned int fe_order;
    unsigned int quadrature_order;
    bool         absorbing_boundary;

    void parse_parameters_callback();
    void make_grid();
    void setup_system();
    void assemble_system(unsigned int t);
    void solve();
    void output_results(unsigned int t);//

    Parameters<dim>            parameters;
    PerfectlyMatchedLayer<dim> perfectly_matched_layer;

    Triangulation<dim> triangulation;
    DoFHandler<dim>    dof_handler;

    std::unique_ptr<FiniteElement<dim>> fe;

    AffineConstraints<double> constraints;
    SparsityPattern           sparsity_pattern;
    SparseMatrix<double>      system_matrix;
    Vector<double>            solution;
    Vector<double>            system_rhs;
  };


  template <int dim>
  Maxwell<dim>::Maxwell()
    : ParameterAcceptor("Maxwell")
    , dof_handler(triangulation)
  {
    ParameterAcceptor::parse_parameters_call_back.connect(
      [&]() { parse_parameters_callback(); });

    scaling = 2;
    add_parameter("scaling", scaling, "scale of the hypercube geometry");

    refinements = 8;
    add_parameter("refinements",
                  refinements,
                  "number of refinements of the geometry");

    fe_order = 0;
    add_parameter("fe order", fe_order, "order of the finite element space");

    quadrature_order = 1;
    add_parameter("quadrature order",
                  quadrature_order,
                  "order of the quadrature");

    absorbing_boundary = true;
    add_parameter("absorbing boundary condition",
                  absorbing_boundary,
                  "use absorbing boundary conditions?");
  }


  template <int dim>
  void Maxwell<dim>::parse_parameters_callback()
  {
    fe = std::make_unique<FESystem<dim>>(FE_NedelecSZ<dim>(fe_order), 2);
  }


  template <int dim>
  void Maxwell<dim>::make_grid()
  {
//    GridGenerator::hyper_cube(triangulation, -scaling, scaling);
//    triangulation.refine_global(refinements);
//    GridGenerator::plate_with_a_hole(
//        triangulation,
//        0.15,                     // inner_radius
//        1.0,                     // outer_radius
//        1.0, 1.0, 1.0, 1.0,      // padding
//        Point<2>(0.0, 0.0),      // hole center
//        0,                        // polar_manifold_id
//        1,                        // tfi_manifold_id
//        1.0,                      // L
//        16,                       // n_slices (关键参数)
//        true                      // colorize
//    );
//    triangulation.refine_global(refinements);
	  // 1. 生成初始网格
	    Triangulation<dim> input_triangulation;
	    GridGenerator::subdivided_hyper_rectangle(
	        input_triangulation,
	        {5, 5},
	        Point<2>(-2.0, -2.0),
	        Point<2>(2.0, 2.0)
	    );
	    input_triangulation.refine_global(6);
	    // 2. 使用 set 存储待移除单元
	    std::set<typename Triangulation<dim>::active_cell_iterator> cells_to_remove;
	    for (const auto &cell : input_triangulation.active_cell_iterators()) {
	        Point<2> center = cell->center();
	        if (center[0] > -0.15 && center[0] < 0.15 &&
	            center[1] > -0.15 && center[1] < 0.15) {
	            cells_to_remove.insert(cell);  // 插入到 set 中
	        }
	    }


	    GridGenerator::create_triangulation_with_removed_cells(
	        input_triangulation,
	        cells_to_remove,  // 传入 set 而非 vector
	        triangulation
	    );

        // 标记材料ID
        for (auto &cell : triangulation.active_cell_iterators()) {
            Point<dim> center = cell->center();
            if(parameters.is_in_hard_source_region(center))
            {
            cell->set_material_id(1);
            }

            else cell->set_material_id(2);


        }

//	    for (auto &face : triangulation.active_face_iterators()) {
//	        if (face->at_boundary()) {
//	            const Point<dim> center = face->center();
//
//	            const double hole_radius = 0.15;  // 孔洞半径
//	            const Point<dim> hole_center(0.0, 0.0); // 孔洞中心坐标（假设在原点）
//
//	            // 计算面中心到孔洞中心的距离
//	            const double distance = center.distance(hole_center);
//                     //   std::cout<<"center"<<distance<<std::endl;
//	            // 标记孔洞边界（考虑浮点误差）
//	            if (std::abs(distance - hole_radius) <= 1e-2) {
//	                face->set_boundary_id(2);  // 设置孔洞边界ID为2（PEC边界）
//
//	            }
//	        }
//	    }

//    }
//std::cout<<"count="<<cs<<std::endl;
//    for (auto &cell : triangulation.active_cell_iterators())
//      if (cell->center()[1] > 0.)
//        cell->set_material_id(1);
//      else
//        cell->set_material_id(2);


    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl;
  }


  template <int dim>
  void Maxwell<dim>::setup_system()
  {
    dof_handler.distribute_dofs(*fe);
    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.clear();

    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    VectorTools::project_boundary_values_curl_conforming_l2(
      dof_handler,
      0, /* real part */
      Functions::ZeroFunction<dim>(2 * dim),
      1, /* boundary id */
      constraints);
    VectorTools::project_boundary_values_curl_conforming_l2(
      dof_handler,
      dim, /* imaginary part */
      Functions::ZeroFunction<dim>(2 * dim),
      1, /* boundary id */
      constraints);


    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /* keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
  }

  template <int dim>
  DEAL_II_ALWAYS_INLINE inline Tensor<1, dim, std::complex<double>>
  tangential_part(const Tensor<1, dim, std::complex<double>> &tensor,
                  const Tensor<1, dim>                       &normal)
  {
    auto result = tensor;
    result[0]   = normal[1] * (tensor[0] * normal[1] - tensor[1] * normal[0]);
    result[1]   = -normal[0] * (tensor[0] * normal[1] - tensor[1] * normal[0]);
    return result;
  }



  template <int dim>
  void Maxwell<dim>::assemble_system(unsigned int t)
  {
    const QGauss<dim>     quadrature_formula(quadrature_order);
    const QGauss<dim - 1> face_quadrature_formula(quadrature_order);

    FEValues<dim, dim>     fe_values(*fe,
                                 quadrature_formula,
                                 update_values | update_gradients |
                                   update_quadrature_points |
                                   update_JxW_values);
    FEFaceValues<dim, dim> fe_face_values(*fe,
                                          face_quadrature_formula,
                                          update_values | update_gradients |
                                            update_quadrature_points |
                                            update_normal_vectors |
                                            update_JxW_values);

    const unsigned int dofs_per_cell = fe->dofs_per_cell;

    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector real_part(0);
    const FEValuesExtractors::Vector imag_part(dim);

	double omega = 14.2;
       int temp=0;
       auto imag = std::complex<double>(0,1);
       const double penalty_parameter = 1e3; // 罚参数，需足够大
       const double penalty_parameter_1 = 0.0;
    const double tf_min = -1, tf_max = 1; // 总场区范围
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);

        cell_matrix = 0.;
        cell_rhs    = 0.;

        cell->get_dof_indices(local_dof_indices);
        const auto id = cell->material_id();
        const bool is_scattered_region = (cell->material_id() == 2); // 散射场区单元


        for (const auto &face : cell->face_iterators())
        {
         fe_face_values.reinit(cell, face);
         for (unsigned int q = 0; q < n_face_q_points; ++q)
         {
        	   const Point<dim> position = fe_face_values.quadrature_point(q);
        	   const bool on_tfsf_boundary = parameters.is_on_tfsf_boundary(position);
        	   if (is_scattered_region )
        	   {
        		   //const auto face_index = cell->face_iterator_to_index(face);
//        		   auto neighbor = cell->neighbor(face_index);
//       			   std::cout<<"neighbor="<<neighbor<<std::endl;
        		   auto normal = parameters.get_boundary_normal(position); // 指向总场区
        		   auto E_inc = parameters.hard_source_E(position, t);
        		   auto E_inc_tangent = E_inc - (E_inc * normal) * normal; // 切向投影
        		   auto  mu_inv  = parameters.mu_inv(position, id);
        		   const auto d = perfectly_matched_layer.d(position);
        		   auto  mu_inv_r = parameters.mu_inv_r(position, id);

        		   constexpr std::complex<double> imag{0., 1.};
        		   mu_inv=mu_inv*mu_inv_r/d;
        		   // 扣除入射场贡献（注意符号）
        		   for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
        		   {
        		     const auto phi_i =
                          	fe_face_values[real_part].value(i, q)
                          -imag * fe_face_values[imag_part].value(i, q);
                    auto phi_i_tangent = phi_i - (phi_i * normal) * normal;// 基函数切向分量
//                    auto temp=(mu_inv * (phi_i_tangent * E_inc_tangent)) * fe_face_values.JxW(q);
//        		    cell_rhs(i) -= temp.real();
//        		    std::cout<<"temp="<<temp<<std::endl;

        		 //  if (neighbor->material_id() == 1) {

        			   for (unsigned int j = 0; j < dofs_per_cell; ++j) {
        			           const auto phi_j = fe_face_values[real_part].value(j, q)
        			                            + imag * fe_face_values[imag_part].value(j, q);
        			           const auto phi_j_tangent = tangential_part(phi_j, normal);
        			       auto     temp = penalty_parameter * scalar_product(phi_i_tangent, phi_j_tangent)
        			                               * fe_face_values.JxW(q);
        			            cell_matrix(i, j) +=temp.real();
        			       }
        		 //  }
                  }
        	   }
        	   if (is_scattered_region && parameters.in_tfsf_boundary_lr(position))
        	   {
        		  // const auto face_index = cell->face_iterator_to_index(face);
//        		   auto neighbor = cell->neighbor(face_index);
//       			   std::cout<<"neighbor="<<neighbor<<std::endl;
        		   auto normal = parameters.get_boundary_normal(position); // 指向总场区
        		   auto E_inc = parameters.hard_source_E(position, t);
        		   auto E_inc_tangent = E_inc - (E_inc * normal) * normal; // 切向投影
        		   auto  mu_inv  = parameters.mu_inv(position, id);
        		   auto  mu_inv_r = parameters.mu_inv_r(position, id);
        		   const auto d = perfectly_matched_layer.d(position);
        		   constexpr std::complex<double> imag{0., 1.};
        		   mu_inv=mu_inv*mu_inv_r/d;
        		   // 扣除入射场贡献（注意符号）
        		   for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
        		   {
        		     const auto phi_i =
                          	fe_face_values[real_part].value(i, q)
                          -imag * fe_face_values[imag_part].value(i, q);
                    auto phi_i_tangent = phi_i - (phi_i * normal) * normal;// 基函数切向分量
//                    auto temp=(mu_inv * (phi_i_tangent * E_inc_tangent)) * fe_face_values.JxW(q);
//        		    cell_rhs(i) -= temp.real();
//        		    std::cout<<"temp="<<temp<<std::endl;

        		 //  if (neighbor->material_id() == 1) {

        			   for (unsigned int j = 0; j < dofs_per_cell; ++j) {
        			           const auto phi_j = fe_face_values[real_part].value(j, q)
        			                            + imag * fe_face_values[imag_part].value(j, q);
        			           const auto phi_j_tangent = tangential_part(phi_j, normal);
        			       auto     temp = penalty_parameter_1 * scalar_product(phi_i_tangent, phi_j_tangent)
        			                               * fe_face_values.JxW(q);
        			            cell_matrix(i, j) +=temp.real();
        			       }
        		 //  }
                  }
        	   }
         }
        }
       // std::cout<<"id="<<id<<std::endl;

//if(id==1){
//	Point<dim> center = cell->center();
//	std::cout<<center<<std::endl;
//}

        const auto &quadrature_points = fe_values.get_quadrature_points();

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            const Point<dim> &position = fe_values.quadrature_point(q_point);
         //   const Point<dim> &position = quadrature_points[q_point];
//            std::cout << "At point " << position
//                      << ": in_pml=" << perfectly_matched_layer.is_in_pml(position)
//                      << ", sigma_x=" << perfectly_matched_layer.compute_sigma_x(position)
//                      << ", sigma_y=" << perfectly_matched_layer.compute_sigma_y(position)
//                      << std::endl;
            auto  mu_inv  = parameters.mu_inv(position, id);
            auto  mu_inv_r = parameters.mu_inv_r(position, id);
            auto  epsilon = parameters.epsilon(position, id);
            auto  epsilon_r = parameters.epsilon_r(position, id);
            const auto J_a     = parameters.J_a(position, id);
//            std::cout<<"d="<<epsilon<<std::endl;
           // const auto A = perfectly_matched_layer.a_matrix(position);
          //  const auto B = perfectly_matched_layer.b_matrix(position);
            const auto d = perfectly_matched_layer.d(position);
            const auto d_bar= perfectly_matched_layer.d_bar(position);
//            std::cout<<"d="<<d<<std::endl;

                    	//  std::cout<<"mu_inv_r="<<id<<std::endl;

            mu_inv  =mu_inv*mu_inv_r/d;
//            if(perfectly_matched_layer.is_in_pml(position)){
//            	if(position[0]<-1.0)
//                       std::cout<<"d="<<d<<std::endl;
//                        }
            //mu_inv=1./mu_inv;
            epsilon = epsilon * epsilon_r;
            epsilon[0][0]= epsilon[0][0]*d_bar;
            epsilon[1][1]= epsilon[1][1]/d_bar;
//           if(parameters.in_material_region(position)){
//            	std::cout<<mu_inv_r<<","<<position<<std::endl;
//            }

//            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
//            	auto face = cell->face(f);
//               Point<dim> center = face->center();
//              if(id==2){
//	          if(parameters.in_tfsf_boundary(center))
//	            {
//                 std::cout<<center<<std::endl;
//	             }
//                 }
//            }



            for (const auto i : fe_values.dof_indices())
              {

                constexpr std::complex<double> imag{0., 1.};

                const auto phi_i =
                  fe_values[real_part].value(i, q_point) -
                  imag * fe_values[imag_part].value(i, q_point);
                const auto curl_phi_i =
                  fe_values[real_part].curl(i, q_point) -
                  imag * fe_values[imag_part].curl(i, q_point);
          //     std::cout<<"phi_i_value"<<phi_i<<std::endl;
               const auto rhs_value =
                  (imag * omega * scalar_product(J_a, phi_i)) * fe_values.JxW(q_point);
               cell_rhs(i) = 0.0*rhs_value.real();
                for (const auto j : fe_values.dof_indices())
                  {
                    const auto phi_j =
                      fe_values[real_part].value(j, q_point) +
                      imag * fe_values[imag_part].value(j, q_point);
                    const auto curl_phi_j =
                      fe_values[real_part].curl(j, q_point) +
                      imag * fe_values[imag_part].curl(j, q_point);

                        const auto temp =
                           (scalar_product(mu_inv * curl_phi_j, curl_phi_i) -
                         		  omega*omega*scalar_product(epsilon * phi_j, phi_i)) *
                           fe_values.JxW(q_point);
                         cell_matrix(i, j) += temp.real();

//                    if(i==1&&q_point==0){
//                    	std::cout<<"j="<<j<<"position="<<position<<"phi_j="<<phi_j<<std::endl;
//
//                    }

                  }
              }


          }
       const FEValuesExtractors::Vector real_part(0);
        const FEValuesExtractors::Vector imag_part(dim);
        for (const auto &face : cell->face_iterators()) {

            Point<dim> center = face->center();
//            if(face->at_boundary())
//                                 {
//                               	  auto id =face->at_boundary();
//
//                               	  std::cout<<"id="<<id<<center<<std::endl;
//                                 }
//                	std::cout<<"is_tfsf_boundary="<<center<<std::endl;
                    fe_face_values.reinit(cell, face);
                    for (unsigned int q = 0; q < n_face_q_points; ++q) {
                        const auto &position = fe_face_values.quadrature_point(q);
                       // auto normal = fe_face_values.normal_vector(q);
                         auto normal = parameters.get_boundary_normal(position);
                         auto normal_s=-parameters.get_boundary_normal(position);
                         if (parameters.in_tfsf_boundary(position)) {
//                        auto curl_E_inc = parameters.curl_incident_field(position, t);
                        constexpr std::complex<double> imag{0., 1.};
                        auto mu_inv = parameters.mu_inv(position, cell->material_id());
                        auto mu_inv_r = parameters.mu_inv_r(position, cell->material_id());
                       auto d= perfectly_matched_layer.d(position);
                       mu_inv  =mu_inv * mu_inv_r/d;
                        // 正确的旋度计算
                       auto E_inc_z = parameters.hard_source_E(position,t);
                       auto curl_E_inc_z = parameters.curl_incident_field(position, t);
                       auto E_inc_tangential = tangential_part(E_inc_z, normal_s);
//                       if(position[0]>0&&position[1]>0&&position[1]<1.0){
//                       std::cout<<"n="<<normal<<",position="<<position<<std::endl;
//                       }
                       // n × (∇×E) = (n_y (∇×E)_z, -n_x (∇×E)_z)

//
//                          const unsigned int edge = cell->face(face)->boundary_id(); // 获取边编号
//                          const Point<dim> vertex0 = cell->vertex(edge);
//                          const Point<dim> vertex1 = cell->vertex((edge + 1) % cell->n_vertices());
//
//                          tangent = vertex1 - vertex0;
//                          tangent /= tangent.norm(); // 单位化


                        for (unsigned int i = 0; i < fe->dofs_per_cell; ++i) {
                            const auto phi_i =
                                              	fe_face_values[real_part].value(i, q)
                                              -imag * fe_face_values[imag_part].value(i, q);
                //            std::cout<<"phi_i_face_value"<<phi_i<<std::endl;
                            auto phi_i_tangential = tangential_part(phi_i,normal_s);
                            auto n_cross_curl =
                                                       normal[1] * curl_E_inc_z * phi_i[0] -
                                                       normal[0] * curl_E_inc_z * phi_i[1];

//                            std::cout<<"size="<<normal<<" E_inc"<<curl_E_inc_z<<std::endl;
                            for (unsigned int j = 0; j < fe->dofs_per_cell; ++j) {

                            }
                            auto temp = (mu_inv * n_cross_curl )* fe_face_values.JxW(q);

                            cell_rhs(i) += temp.real();
                       //   std::cout<<"temp="<<phi_i_tangential*E_inc_tangential*fe_face_values.JxW(q)<<std::endl;
                          // std::cout<<"curl_E_inc="<<position<<",NORMAL="<<normal<<std::endl;

                        }
                         }
                    }
            }


        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }
  }

  template <int dim>
  void Maxwell<dim>::solve()
  {
    SparseDirectUMFPACK A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult(solution, system_rhs);

    double tempt = 0;
for (int i=0;i<system_rhs.size();++i){
	tempt += system_rhs(i);

}
std::cout<< tempt <<std::endl;
  }
  // 定义自定义函数类
  template <int dim>
  class HardSourceFunction : public Function<dim>
  {
  public:
    HardSourceFunction(const Step81::Parameters<dim> &params, double t)
      : Function<dim>(4) // 关键：声明输出分量为4个（实部x、y + 虚部x、y）
      , parameters(params)
      , t(t)
    {}

    virtual double value(const Point<dim> &p,
                         const unsigned int component) const override
    {
      // 分量0: 实部x，分量1: 实部y
      // 分量2: 虚部x，分量3: 虚部y
      if (parameters.is_in_inc_source_region(p))
      {
        const auto E_inc = parameters.hard_source_E(p, t);
        if (component == 0)
          return E_inc[0].real(); // 实部x
        else if (component == 1)
          return E_inc[1].real(); // 实部y
        else if (component == 2)
          return E_inc[0].imag(); // 虚部x
        else if (component == 3)
          return E_inc[1].imag(); // 虚部y
      }
      return 0.0; // 其他分量或区域外返回0
    }

  private:
    const Step81::Parameters<dim> &parameters;
    double t;
  };

  template <int dim>
  void Maxwell<dim>::output_results(unsigned int t)
  {
	  DataOut<dim> data_out;
	  data_out.attach_dof_handler(dof_handler);



	   // 创建自定义函数对象
	     HardSourceFunction<dim> hard_source(parameters, t);


	     // 投影到有限元空间
	     Vector<double> incident_field_real(dof_handler.n_dofs());
	     VectorTools::project(
	       dof_handler,
	       constraints,
	       QGauss<dim>(fe->degree + 1),
	       hard_source, // 传入 Function 对象
	       incident_field_real
	     );
//		  // 定义散射场的分量名称（添加前缀 "scattered_"）
//		  std::vector<std::string> scattered_component_names;
//		  scattered_component_names.emplace_back("scattered_E_real_x");
//		  scattered_component_names.emplace_back("scattered_E_real_y");
//		  if (dim == 3)
//		    scattered_component_names.emplace_back("scattered_E_real_z");
//		  scattered_component_names.emplace_back("scattered_E_imag_x");
//		  scattered_component_names.emplace_back("scattered_E_imag_y");
//		  if (dim == 3)
//		    scattered_component_names.emplace_back("scattered_E_imag_z");
//		  Vector<double> scattered_field = solution;
//
//		  scattered_field -= incident_field_real;
//		  // 输出散射场
//		  data_out.add_data_vector( scattered_field, scattered_component_names);
	   // 计算总场：入射场（实部） + 散射场（实部）
	   Vector<double> total_field = solution;
	 // total_field += incident_field_real;

	   // 定义总场的分量名称（添加前缀 "total_"）
	    std::vector<std::string> total_component_names;
	    total_component_names.emplace_back("total_E_real_x");
	    total_component_names.emplace_back("total_E_real_y");
	    if (dim == 3)
	      total_component_names.emplace_back("total_E_real_z");
	    total_component_names.emplace_back("total_E_imag_x");
	    total_component_names.emplace_back("total_E_imag_y");
	    if (dim == 3)
	      total_component_names.emplace_back("total_E_imag_z");

	    // 输出总场
	    data_out.add_data_vector(total_field, total_component_names);
		   Vector<float> pml_flag(triangulation.n_active_cells());
		   for (const auto &cell : triangulation.active_cell_iterators()) {
		       pml_flag[cell->active_cell_index()] =
		           perfectly_matched_layer.is_in_pml(cell->center()) ? 0.5 : 0.0;
		   }
		   data_out.add_data_vector(pml_flag, "PML_Flag");

	   data_out.build_patches();
	   std::ofstream output("solution-" + Utilities::int_to_string(t) + ".vtu");
	   data_out.write_vtu(output);
	   // 在output_results中添加


  }




   template <int dim>
   void Maxwell<dim>::run()
   {
	    make_grid();
	    setup_system();
	   // 时间步进参数
	    double t =0.0;
int step=1;
	      std::cout << "Time step: " << step << ", Time: " << t << std::endl;

	      // 组装系统（传入当前时间 t）
	      assemble_system(t);

	      // 求解系统
	      solve();

	      // 输出结果
	      output_results(t);


       }

}

   int main()
   {
     try
       {
         using namespace dealii;

         Step81::Maxwell<2> maxwell_2d;
         ParameterAcceptor::initialize("parameters.prm");
         maxwell_2d.run();
       }
     catch (std::exception &exc)
       {
         std::cerr << std::endl
                   << std::endl
                   << "----------------------------------------------------"
                   << std::endl;
         std::cerr << "Exception on processing: " << std::endl
                   << exc.what() << std::endl
                   << "Aborting!" << std::endl
                   << "----------------------------------------------------"
                   << std::endl;
         return 1;
       }
     catch (...)
       {
         std::cerr << std::endl
                   << std::endl
                   << "----------------------------------------------------"
                   << std::endl;
         std::cerr << "Unknown exception!" << std::endl
                   << "Aborting!" << std::endl
                   << "----------------------------------------------------"
                   << std::endl;
         return 1;
       }
     return 0;
   }
