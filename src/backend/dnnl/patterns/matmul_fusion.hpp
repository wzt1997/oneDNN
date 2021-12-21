/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#ifndef BACKEND_DNNL_PATTERNS_MATMUL_FUSION_HPP
#define BACKEND_DNNL_PATTERNS_MATMUL_FUSION_HPP

#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "backend/dnnl/patterns/transformation_pattern.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pass {

using pattern = impl::pass::pattern;
using FCreatePattern = impl::pass::FCreatePattern;
using FCreateOptPattern = impl::pass::FCreateOptPattern;

/*!
 * \brief This provides matmul-related fusion, i.e.
 *        matmul-relu fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(matmul_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_relu_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    relu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_elu_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *elu = apattern->create_op(impl::op_kind::Elu);
                    elu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op
                            = optimized_pattern->create_op(op_kind::matmul_elu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_sigmoid_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    sigmoid->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_hardtanh_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *hardtanh
                            = apattern->create_op(impl::op_kind::HardTanh);
                    hardtanh->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_hardtanh);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_gelu_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    gelu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_gelu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_div_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *div = apattern->create_op(impl::op_kind::Divide);
                    div->fill_and_connect_input(0, *matmul, 0);
                    div->fill_and_connect_input(1, *wildcard, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op
                            = optimized_pattern->create_op(op_kind::matmul_div);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_div_add_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *div = apattern->create_op(impl::op_kind::Divide);
                    div->fill_and_connect_input(0, *matmul, 0);
                    div->fill_and_connect_input(1, *wildcard, 0);
                    op_t *wildcard2
                            = apattern->create_op(impl::op_kind::Wildcard);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    add->fill_and_connect_input(0, *div, 0);
                    add->fill_and_connect_input(1, *wildcard2, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_div_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_fusion)
        .set_priority(8.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    bias->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_sigmoid_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    sigmoid->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    sigmoid->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_swish_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    op_t *mul = apattern->create_op(impl::op_kind::Multiply);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    sigmoid->fill_and_connect_input(0, *bias, 0);
                    mul->fill_and_connect_input(0, *bias, 0);
                    mul->fill_and_connect_input(1, *sigmoid, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    op_t *mul = apattern->create_op(impl::op_kind::Multiply);
                    sigmoid->fill_and_connect_input(0, *matmul, 0);
                    mul->fill_and_connect_input(0, *matmul, 0);
                    mul->fill_and_connect_input(1, *sigmoid, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_swish);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_elu_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *elu = apattern->create_op(impl::op_kind::Elu);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    elu->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *elu = apattern->create_op(impl::op_kind::Elu);
                    elu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_elu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_relu_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    relu->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    relu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_relu6_fusion)
        .set_priority(9.1f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *relu6 = apattern->create_op(impl::op_kind::HardTanh);
                    relu6->set_attr<float>("min", 0);
                    relu6->set_attr<float>("max", 6);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    relu6->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *relu6 = apattern->create_op(impl::op_kind::HardTanh);
                    relu6->set_attr<float>("min", 0);
                    relu6->set_attr<float>("max", 6);
                    relu6->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_relu6);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_gelu_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    gelu->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    gelu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_gelu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_hardtanh_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *hardtanh
                            = apattern->create_op(impl::op_kind::HardTanh);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    hardtanh->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *hardtanh
                            = apattern->create_op(impl::op_kind::HardTanh);
                    hardtanh->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_hardtanh);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_matmul_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    quant->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *typecast_data
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_weight
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_dst
                            = apattern->create_op(impl::op_kind::TypeCast);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    typecast_data->set_attr<bool>("out_bf16_check", true);
                    typecast_weight->set_attr<bool>("out_bf16_check", true);
                    typecast_dst->set_attr<bool>("in_bf16_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    typecast_data->fill_and_connect_input(0, *dequant_data, 0);
                    typecast_weight->fill_and_connect_input(
                            0, *dequant_weight, 0);
                    matmul->fill_and_connect_input(0, *typecast_data, 0);
                    matmul->fill_and_connect_input(1, *typecast_weight, 0);
                    typecast_dst->fill_and_connect_input(0, *matmul, 0);
                    quant->fill_and_connect_input(0, *typecast_dst, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_matmul);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_quant_wei_matmul_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    quant->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_quant_wei_matmul);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_matmul_bias_fusion)
        .set_priority(10.5f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    quant->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    quant->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *typecast_data
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_weight
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_dst
                            = apattern->create_op(impl::op_kind::TypeCast);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    typecast_data->set_attr<bool>("out_bf16_check", true);
                    typecast_weight->set_attr<bool>("out_bf16_check", true);
                    typecast_dst->set_attr<bool>("in_bf16_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    typecast_data->fill_and_connect_input(0, *dequant_data, 0);
                    typecast_weight->fill_and_connect_input(
                            0, *dequant_weight, 0);
                    matmul->fill_and_connect_input(0, *typecast_data, 0);
                    matmul->fill_and_connect_input(1, *typecast_weight, 0);
                    typecast_dst->fill_and_connect_input(0, *matmul, 0);
                    quant->fill_and_connect_input(0, *typecast_dst, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_matmul_bias);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_quant_wei_matmul_bias_fusion)
        .set_priority(10.6f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    quant->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    quant->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_quant_wei_matmul_bias);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_matmul_relu_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    relu->fill_and_connect_input(0, *matmul, 0);
                    quant->fill_and_connect_input(0, *relu, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_matmul_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_quant_wei_matmul_relu_fusion)
        .set_priority(10.6f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    relu->fill_and_connect_input(0, *matmul, 0);
                    quant->fill_and_connect_input(0, *relu, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_quant_wei_matmul_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_matmul_bias_relu_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    relu->fill_and_connect_input(0, *matmul, 0);
                    quant->fill_and_connect_input(0, *relu, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    relu->fill_and_connect_input(0, *bias, 0);
                    quant->fill_and_connect_input(0, *relu, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_matmul_bias_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_quant_wei_matmul_bias_relu_fusion)
        .set_priority(10.6f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    relu->fill_and_connect_input(0, *matmul, 0);
                    quant->fill_and_connect_input(0, *relu, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    relu->fill_and_connect_input(0, *bias, 0);
                    quant->fill_and_connect_input(0, *relu, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_quant_wei_matmul_bias_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_matmul_sigmoid_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    sigmoid->fill_and_connect_input(0, *matmul, 0);
                    quant->fill_and_connect_input(0, *sigmoid, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_matmul_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_quant_wei_matmul_sigmoid_fusion)
        .set_priority(10.6f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    sigmoid->fill_and_connect_input(0, *matmul, 0);
                    quant->fill_and_connect_input(0, *sigmoid, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_quant_wei_matmul_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_matmul_bias_sigmoid_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    sigmoid->fill_and_connect_input(0, *matmul, 0);
                    quant->fill_and_connect_input(0, *sigmoid, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    sigmoid->fill_and_connect_input(0, *bias, 0);
                    quant->fill_and_connect_input(0, *sigmoid, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_matmul_bias_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_quant_wei_matmul_bias_sigmoid_fusion)
        .set_priority(10.6f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    sigmoid->fill_and_connect_input(0, *matmul, 0);
                    quant->fill_and_connect_input(0, *sigmoid, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    sigmoid->fill_and_connect_input(0, *bias, 0);
                    quant->fill_and_connect_input(0, *sigmoid, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_quant_wei_matmul_bias_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_matmul_gelu_fusion)
        .set_priority(9.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    gelu->fill_and_connect_input(0, *matmul, 0);
                    quant->fill_and_connect_input(0, *gelu, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *typecast_data
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_weight
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_gelu
                            = apattern->create_op(impl::op_kind::TypeCast);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    typecast_data->set_attr<bool>("out_bf16_check", true);
                    typecast_weight->set_attr<bool>("out_bf16_check", true);
                    typecast_gelu->set_attr<bool>("in_bf16_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    typecast_data->fill_and_connect_input(0, *dequant_data, 0);
                    typecast_weight->fill_and_connect_input(
                            0, *dequant_weight, 0);
                    matmul->fill_and_connect_input(0, *typecast_data, 0);
                    matmul->fill_and_connect_input(1, *typecast_weight, 0);
                    gelu->fill_and_connect_input(0, *matmul, 0);
                    typecast_gelu->fill_and_connect_input(0, *gelu, 0);
                    quant->fill_and_connect_input(0, *typecast_gelu, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_matmul_gelu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_quant_wei_matmul_gelu_fusion)
        .set_priority(10.6f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    gelu->fill_and_connect_input(0, *matmul, 0);
                    quant->fill_and_connect_input(0, *gelu, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_quant_wei_matmul_gelu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_matmul_bias_gelu_fusion)
        .set_priority(10.5f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    gelu->fill_and_connect_input(0, *matmul, 0);
                    quant->fill_and_connect_input(0, *gelu, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    gelu->fill_and_connect_input(0, *bias, 0);
                    quant->fill_and_connect_input(0, *gelu, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *typecast_data
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_weight
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_gelu
                            = apattern->create_op(impl::op_kind::TypeCast);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    typecast_data->set_attr<bool>("out_bf16_check", true);
                    typecast_weight->set_attr<bool>("out_bf16_check", true);
                    typecast_gelu->set_attr<bool>("in_bf16_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    typecast_data->fill_and_connect_input(0, *dequant_data, 0);
                    typecast_weight->fill_and_connect_input(
                            0, *dequant_weight, 0);
                    matmul->fill_and_connect_input(0, *typecast_data, 0);
                    matmul->fill_and_connect_input(1, *typecast_weight, 0);
                    gelu->fill_and_connect_input(0, *matmul, 0);
                    typecast_gelu->fill_and_connect_input(0, *gelu, 0);
                    quant->fill_and_connect_input(0, *typecast_gelu, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_matmul_bias_gelu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_quant_wei_matmul_bias_gelu_fusion)
        .set_priority(10.6f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    gelu->fill_and_connect_input(0, *matmul, 0);
                    quant->fill_and_connect_input(0, *gelu, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    gelu->fill_and_connect_input(0, *bias, 0);
                    quant->fill_and_connect_input(0, *gelu, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_quant_wei_matmul_bias_gelu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8x8f32_matmul_fusion)
        .set_priority(9.5f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8x8float_matmul);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8f32_matmul_bias_fusion)
        .set_priority(9.6f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8float_matmul_bias);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8f32_matmul_relu_fusion)
        .set_priority(9.6f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    relu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8f32_matmul_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8f32_matmul_bias_relu_fusion)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    relu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    relu->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8f32_matmul_bias_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8f32_matmul_sigmoid_fusion)
        .set_priority(9.6f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    sigmoid->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8f32_matmul_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8f32_matmul_bias_sigmoid_fusion)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    sigmoid->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    sigmoid->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8f32_matmul_bias_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8f32_matmul_gelu_fusion)
        .set_priority(9.6f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    gelu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8f32_matmul_gelu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8f32_matmul_bias_gelu_fusion)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    gelu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    gelu->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8f32_matmul_bias_gelu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8f32_quant_wei_matmul_fusion)
        .set_priority(9.6f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8f32_quant_wei_matmul);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8f32_quant_wei_matmul_bias_fusion)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8f32_quant_wei_matmul_bias);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8f32_quant_wei_matmul_relu_fusion)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    relu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8f32_quant_wei_matmul_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8f32_quant_wei_matmul_bias_relu_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    relu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    relu->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8f32_quant_wei_matmul_bias_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8f32_quant_wei_matmul_sigmoid_fusion)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    sigmoid->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8f32_quant_wei_matmul_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8f32_quant_wei_matmul_bias_sigmoid_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    sigmoid->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    sigmoid->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8f32_quant_wei_matmul_bias_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8f32_quant_wei_matmul_gelu_fusion)
        .set_priority(9.7f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    gelu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8f32_quant_wei_matmul_gelu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8f32_quant_wei_matmul_bias_gelu_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    gelu->fill_and_connect_input(0, *matmul, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    gelu->fill_and_connect_input(0, *bias, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8f32_quant_wei_matmul_bias_gelu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_matmul_add_fusion)
        .set_priority(10.5f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *dequant_other
                            = apattern->create_op(impl::op_kind::Dequantize);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *dequant_other, 0);
                    quant->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_matmul_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_quant_wei_matmul_add_fusion)
        .set_priority(10.6f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *dequant_other
                            = apattern->create_op(impl::op_kind::Dequantize);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *dequant_other, 0);
                    quant->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_quant_wei_matmul_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_matmul_bias_add_fusion)
        .set_priority(10.5f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *dequant_other
                            = apattern->create_op(impl::op_kind::Dequantize);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *dequant_other, 0);
                    quant->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    op_t *dequant_other
                            = apattern->create_op(impl::op_kind::Dequantize);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(0, *bias, 0);
                    add->fill_and_connect_input(1, *dequant_other, 0);
                    quant->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_matmul_bias_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_quant_wei_matmul_bias_add_fusion)
        .set_priority(10.6f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *dequant_other
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *dequant_other, 0);
                    quant->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *dequant_other
                            = apattern->create_op(impl::op_kind::Dequantize);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    op_t *quant = apattern->create_op(impl::op_kind::Quantize);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(0, *bias, 0);
                    add->fill_and_connect_input(1, *dequant_other, 0);
                    quant->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::int8_quant_wei_matmul_bias_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8f32_matmul_add_fusion)
        .set_priority(10.4f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_other
                            = apattern->create_op(impl::op_kind::Dequantize);

                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *dequant_other, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8float_matmul_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8x8f32_matmul_div_fusion)
        .set_priority(10.4f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *div = apattern->create_op(impl::op_kind::Divide);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    div->fill_and_connect_input(0, *matmul, 0);
                    div->fill_and_connect_input(1, *wildcard, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8x8float_matmul_div);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8x8f32_matmul_div_add_fusion)
        .set_priority(10.5f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *div = apattern->create_op(impl::op_kind::Divide);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    div->fill_and_connect_input(0, *matmul, 0);
                    div->fill_and_connect_input(1, *wildcard, 0);
                    op_t *wildcard2
                            = apattern->create_op(impl::op_kind::Wildcard);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    add->fill_and_connect_input(0, *div, 0);
                    add->fill_and_connect_input(1, *wildcard2, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8x8float_matmul_div_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8f32_matmul_bias_add_fusion)
        .set_priority(10.4f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_other
                            = apattern->create_op(impl::op_kind::Dequantize);

                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *dequant_other, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_other
                            = apattern->create_op(impl::op_kind::Dequantize);

                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(0, *bias, 0);
                    add->fill_and_connect_input(1, *dequant_other, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8float_matmul_bias_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8f32_quant_wei_matmul_add_fusion)
        .set_priority(10.5f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *dequant_other
                            = apattern->create_op(impl::op_kind::Dequantize);

                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *dequant_other, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8f32_quant_wei_matmul_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8f32_quant_wei_matmul_bias_add_fusion)
        .set_priority(10.5f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *dequant_other
                            = apattern->create_op(impl::op_kind::Dequantize);

                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *dequant_other, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *quant_weight
                            = apattern->create_op(impl::op_kind::Quantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    dequant_weight->fill_and_connect_input(0, *quant_weight, 0);
                    op_t *dequant_other
                            = apattern->create_op(impl::op_kind::Dequantize);

                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    matmul->fill_and_connect_input(0, *dequant_data, 0);
                    matmul->fill_and_connect_input(1, *dequant_weight, 0);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(0, *bias, 0);
                    add->fill_and_connect_input(1, *dequant_other, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8f32_quant_wei_matmul_bias_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_sum_relu_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(0, *bias, 0);
                    add->fill_and_connect_input(1, *wildcard, 0);
                    relu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *wildcard, 0);
                    relu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_bias_sum_fusion)
        .set_priority(9.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *bias = apattern->create_op(impl::op_kind::BiasAdd);
                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    bias->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(0, *bias, 0);
                    add->fill_and_connect_input(1, *wildcard, 0);
                })
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);
                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *wildcard, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_bias_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_sum_fusion)
        .set_priority(8.9f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *wildcard, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op
                            = optimized_pattern->create_op(op_kind::matmul_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_sum_gelu_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    op_t *gelu = apattern->create_op(impl::op_kind::GELU);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *wildcard, 0);
                    gelu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_add_gelu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_sum_relu_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    op_t *relu = apattern->create_op(impl::op_kind::ReLU);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *wildcard, 0);
                    relu->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_add_relu);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_sum_sigmoid_fusion)
        .set_priority(10.0f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);
                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);
                    op_t *add = apattern->create_op(impl::op_kind::Add);
                    op_t *sigmoid = apattern->create_op(impl::op_kind::Sigmoid);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *wildcard, 0);
                    sigmoid->fill_and_connect_input(0, *add, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::matmul_add_sigmoid);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8x8bf16_matmul_fusion)
        .set_priority(9.8f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *typecast_data
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_weight
                            = apattern->create_op(impl::op_kind::TypeCast);
                    // this pattern requires the output dtype to be bf16
                    typecast_data->set_attr<bool>("out_bf16_check", true);
                    typecast_weight->set_attr<bool>("out_bf16_check", true);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);

                    typecast_data->fill_and_connect_input(0, *dequant_data, 0);
                    typecast_weight->fill_and_connect_input(
                            0, *dequant_weight, 0);
                    matmul->fill_and_connect_input(0, *typecast_data, 0);
                    matmul->fill_and_connect_input(1, *typecast_weight, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8x8float_matmul);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8x8bf16_matmul_div_fusion)
        .set_priority(10.4f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *typecast_data
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_weight
                            = apattern->create_op(impl::op_kind::TypeCast);
                    // this pattern requires the output dtype to be bf16
                    typecast_data->set_attr<bool>("out_bf16_check", true);
                    typecast_weight->set_attr<bool>("out_bf16_check", true);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);

                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);
                    op_t *div = apattern->create_op(impl::op_kind::Divide);

                    typecast_data->fill_and_connect_input(0, *dequant_data, 0);
                    typecast_weight->fill_and_connect_input(
                            0, *dequant_weight, 0);
                    matmul->fill_and_connect_input(0, *typecast_data, 0);
                    matmul->fill_and_connect_input(1, *typecast_weight, 0);
                    div->fill_and_connect_input(0, *matmul, 0);
                    div->fill_and_connect_input(1, *wildcard, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8x8float_matmul_div);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8x8bf16_matmul_div_add_fusion)
        .set_priority(10.5f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *typecast_data
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_weight
                            = apattern->create_op(impl::op_kind::TypeCast);
                    // this pattern requires the output dtype to be bf16
                    typecast_data->set_attr<bool>("out_bf16_check", true);
                    typecast_weight->set_attr<bool>("out_bf16_check", true);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);

                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);
                    op_t *div = apattern->create_op(impl::op_kind::Divide);

                    op_t *wildcard2
                            = apattern->create_op(impl::op_kind::Wildcard);
                    op_t *add = apattern->create_op(impl::op_kind::Add);

                    typecast_data->fill_and_connect_input(0, *dequant_data, 0);
                    typecast_weight->fill_and_connect_input(
                            0, *dequant_weight, 0);
                    matmul->fill_and_connect_input(0, *typecast_data, 0);
                    matmul->fill_and_connect_input(1, *typecast_weight, 0);
                    div->fill_and_connect_input(0, *matmul, 0);
                    div->fill_and_connect_input(1, *wildcard, 0);
                    add->fill_and_connect_input(0, *div, 0);
                    add->fill_and_connect_input(1, *wildcard2, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8x8float_matmul_div_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8bf16_matmul_bias_fusion)
        .set_priority(10.4f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *typecast_data
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_weight
                            = apattern->create_op(impl::op_kind::TypeCast);

                    typecast_data->set_attr<bool>("out_bf16_check", true);
                    typecast_weight->set_attr<bool>("out_bf16_check", true);

                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);

                    typecast_data->fill_and_connect_input(0, *dequant_data, 0);
                    typecast_weight->fill_and_connect_input(
                            0, *dequant_weight, 0);
                    matmul->fill_and_connect_input(0, *typecast_data, 0);
                    matmul->fill_and_connect_input(1, *typecast_weight, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8float_matmul_bias);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8bf16_matmul_bias_add_fusion)
        .set_priority(10.5f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_other
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *typecast_data
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_weight
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_other
                            = apattern->create_op(impl::op_kind::TypeCast);

                    typecast_data->set_attr<bool>("out_bf16_check", true);
                    typecast_weight->set_attr<bool>("out_bf16_check", true);
                    typecast_other->set_attr<bool>("out_bf16_check", true);

                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);

                    op_t *add = apattern->create_op(impl::op_kind::Add);

                    typecast_data->fill_and_connect_input(0, *dequant_data, 0);
                    typecast_weight->fill_and_connect_input(
                            0, *dequant_weight, 0);
                    typecast_other->fill_and_connect_input(
                            0, *dequant_other, 0);
                    matmul->fill_and_connect_input(0, *typecast_data, 0);
                    matmul->fill_and_connect_input(1, *typecast_weight, 0);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *typecast_other, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8float_matmul_bias_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, x8s8bf16_matmul_bias_add_bf16_fusion)
        .set_priority(10.49f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *typecast_data
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_weight
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *wildcard
                            = apattern->create_op(impl::op_kind::Wildcard);

                    typecast_data->set_attr<bool>("out_bf16_check", true);
                    typecast_weight->set_attr<bool>("out_bf16_check", true);

                    // this pattern requires the weight should be s8
                    dequant_weight->set_attr<bool>("s8_check", true);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 3);

                    op_t *add = apattern->create_op(impl::op_kind::Add);

                    typecast_data->fill_and_connect_input(0, *dequant_data, 0);
                    typecast_weight->fill_and_connect_input(
                            0, *dequant_weight, 0);
                    matmul->fill_and_connect_input(0, *typecast_data, 0);
                    matmul->fill_and_connect_input(1, *typecast_weight, 0);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *wildcard, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8float_matmul_bias_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, x8s8bf16_matmul_add_fusion)
        .set_priority(10.3f)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](pattern *apattern) -> void {
                    op_t *dequant_data
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_weight
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *dequant_other
                            = apattern->create_op(impl::op_kind::Dequantize);
                    op_t *typecast_data
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_weight
                            = apattern->create_op(impl::op_kind::TypeCast);
                    op_t *typecast_other
                            = apattern->create_op(impl::op_kind::TypeCast);

                    typecast_data->set_attr<bool>("out_bf16_check", true);
                    typecast_weight->set_attr<bool>("out_bf16_check", true);
                    typecast_other->set_attr<bool>("out_bf16_check", true);

                    op_t *matmul = apattern->create_op(impl::op_kind::MatMul);
                    matmul->set_attr<int64_t>("num_inputs", 2);

                    op_t *add = apattern->create_op(impl::op_kind::Add);

                    typecast_data->fill_and_connect_input(0, *dequant_data, 0);
                    typecast_weight->fill_and_connect_input(
                            0, *dequant_weight, 0);
                    typecast_other->fill_and_connect_input(
                            0, *dequant_other, 0);
                    matmul->fill_and_connect_input(0, *typecast_data, 0);
                    matmul->fill_and_connect_input(1, *typecast_weight, 0);
                    add->fill_and_connect_input(0, *matmul, 0);
                    add->fill_and_connect_input(1, *typecast_other, 0);
                })
        .set_attr<FCreateOptPattern>(
                "FCreateOptPattern", [](pattern *optimized_pattern) -> void {
                    op_t *fused_op = optimized_pattern->create_op(
                            op_kind::x8s8float_matmul_add);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_MHA_fusion)
        .set_priority(5.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto query_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "query_reshape");
                    auto query_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, query_reshape, 0)},
                                    "query_transpose");
                    auto quantize_query
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, query_transpose, 0)},
                                    "quantize_query");
                    auto dequantize_query
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quantize_query, 0)},
                                    "dequantize_query");

                    auto key_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "key_reshape");
                    auto key_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, key_reshape, 0)},
                                    "key_transpose");
                    auto key_transpose2
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, key_transpose, 0)},
                                    "key_transpose2");
                    auto quantize_key
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, key_transpose2, 0)},
                                    "quantize_key");
                    auto dequantize_key
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quantize_key, 0)},
                                    "dequantize_key");
                    auto matmul_qk = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, dequantize_query, 0),
                                    in_edge(1, dequantize_key, 0)},
                            "matmul_qk");

                    auto fscore_scale = pgraph->append_op(impl::op_kind::Divide,
                            in_edges_t {in_edge(0, matmul_qk, 0)},
                            "fscore_scale");
                    auto fscore_add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, fscore_scale, 0)},
                            "fscore_add");
                    fscore_add->set_commutative_pair({0, 1});
                    auto softmax = pgraph->append_op(impl::op_kind::SoftMax,
                            in_edges_t {in_edge(0, fscore_add, 0)}, "softmax");
                    auto quantize_softmax
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, softmax, 0)},
                                    "quantize_softmax");
                    auto dequantize_softmax = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, quantize_softmax, 0)},
                            "dequantize_softmax");

                    auto value_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "value_reshape");
                    auto value_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, value_reshape, 0)},
                                    "value_transpose");
                    auto quantize_value
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, value_transpose, 0)},
                                    "quantize_value");
                    auto dequantize_value
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quantize_value, 0)},
                                    "dequantize_value");
                    auto matmul_v = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, dequantize_softmax, 0),
                                    in_edge(1, dequantize_value, 0)},
                            "matmul_v");
                    pgraph->append_op(impl::op_kind::StaticTranspose,
                            in_edges_t {in_edge(0, matmul_v, 0)},
                            "transpose_output");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::int8_MHA);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, f32_MHA_fusion)
        .set_priority(5.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto query_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "query_reshape");
                    auto query_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, query_reshape, 0)},
                                    "query_transpose");

                    auto key_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "key_reshape");
                    auto key_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, key_reshape, 0)},
                                    "key_transpose");
                    auto key_transpose2
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, key_transpose, 0)},
                                    "key_transpose2");
                    auto matmul_qk = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, query_transpose, 0),
                                    in_edge(1, key_transpose2, 0)},
                            "matmul_qk");

                    auto fscore_scale = pgraph->append_op(impl::op_kind::Divide,
                            in_edges_t {in_edge(0, matmul_qk, 0)},
                            "fscore_scale");
                    auto fscore_add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, fscore_scale, 0)},
                            "fscore_add");
                    fscore_add->set_commutative_pair({0, 1});
                    auto softmax = pgraph->append_op(impl::op_kind::SoftMax,
                            in_edges_t {in_edge(0, fscore_add, 0)}, "softmax");

                    auto value_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "value_reshape");
                    auto value_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, value_reshape, 0)},
                                    "value_transpose");

                    auto matmul_v = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, softmax, 0),
                                    in_edge(1, value_transpose, 0)},
                            "matmul_v");
                    pgraph->append_op(impl::op_kind::StaticTranspose,
                            in_edges_t {in_edge(0, matmul_v, 0)},
                            "transpose_output");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::f32_MHA);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_bf16_MHA_fusion)
        .set_priority(5.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto query_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "query_reshape");
                    auto query_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, query_reshape, 0)},
                                    "query_transpose");
                    auto bf16_to_f32_query
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, query_transpose, 0)},
                                    "bf16_to_f32_query");
                    auto quantize_query = pgraph->append_op(
                            impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, bf16_to_f32_query, 0)},
                            "quantize_query");
                    auto dequantize_query
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quantize_query, 0)},
                                    "dequantize_query");
                    auto f32_to_bf16_query = pgraph->append_op(
                            impl::op_kind::TypeCast,
                            in_edges_t {in_edge(0, dequantize_query, 0)},
                            "f32_to_bf16_query");

                    auto key_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "key_reshape");
                    auto key_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, key_reshape, 0)},
                                    "key_transpose");
                    auto key_transpose2
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, key_transpose, 0)},
                                    "key_transpose2");
                    auto bf16_to_f32_key
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, key_transpose2, 0)},
                                    "bf16_to_f32_key");
                    auto quantize_key
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, bf16_to_f32_key, 0)},
                                    "quantize_key");
                    auto dequantize_key
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quantize_key, 0)},
                                    "dequantize_key");
                    auto f32_to_bf16_key
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequantize_key, 0)},
                                    "f32_to_bf16_key");
                    auto matmul_qk = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, f32_to_bf16_query, 0),
                                    in_edge(1, f32_to_bf16_key, 0)},
                            "matmul_qk");

                    auto fscore_scale = pgraph->append_op(impl::op_kind::Divide,
                            in_edges_t {in_edge(0, matmul_qk, 0)},
                            "fscore_scale");
                    auto fscore_add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, fscore_scale, 0)},
                            "fscore_add");
                    fscore_add->set_commutative_pair({0, 1});
                    auto softmax = pgraph->append_op(impl::op_kind::SoftMax,
                            in_edges_t {in_edge(0, fscore_add, 0)}, "softmax");
                    auto bf16_to_f32_softmax
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, softmax, 0)},
                                    "bf16_to_f32_softmax");
                    auto quantize_softmax = pgraph->append_op(
                            impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, bf16_to_f32_softmax, 0)},
                            "quantize_softmax");
                    auto dequantize_softmax = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, quantize_softmax, 0)},
                            "dequantize_softmax");
                    auto f32_to_bf16_softmax = pgraph->append_op(
                            impl::op_kind::TypeCast,
                            in_edges_t {in_edge(0, dequantize_softmax, 0)},
                            "f32_to_bf16_softmax");

                    auto value_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "value_reshape");
                    auto value_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, value_reshape, 0)},
                                    "value_transpose");
                    auto bf16_to_f32_value
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, value_transpose, 0)},
                                    "bf16_to_f32_value");
                    auto quantize_value = pgraph->append_op(
                            impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, bf16_to_f32_value, 0)},
                            "quantize_value");
                    auto dequantize_value
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quantize_value, 0)},
                                    "dequantize_value");
                    auto f32_to_bf16_value = pgraph->append_op(
                            impl::op_kind::TypeCast,
                            in_edges_t {in_edge(0, dequantize_value, 0)},
                            "f32_to_bf16_value");
                    auto matmul_v = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, f32_to_bf16_softmax, 0),
                                    in_edge(1, f32_to_bf16_value, 0)},
                            "matmul_v");
                    pgraph->append_op(impl::op_kind::StaticTranspose,
                            in_edges_t {in_edge(0, matmul_v, 0)},
                            "transpose_output");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::int8_MHA);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
