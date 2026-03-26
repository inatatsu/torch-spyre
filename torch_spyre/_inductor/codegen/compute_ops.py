# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from torch_spyre._C import encode_constant, DataFormats
from sympy import Symbol


def core_idx_to_slice_offset(
    arg,
    wk_slice: dict,
    work_slices: dict,
) -> int:
    offset = 0
    for dim, stride in arg.strides.items():
        if str(dim) in wk_slice and arg.scales[dim] > 0:
            offset += wk_slice[str(dim)] * stride // work_slices[dim]
    return offset


def num_bytes(df: DataFormats) -> int:
    """Try to avoid using this method; it is a bad API due to sub-byte datatypes"""
    num_elems = df.elems_per_stick()
    if num_elems > 128:
        raise RuntimeError(f"sub-byte dataformat {df}")
    return 128 // num_elems


def generate_constant_info(data_format, constants, num_cores):
    if len(constants.keys()) == 0:
        return "{}"
    constant_info = {}
    for name, value in constants.items():
        ci = {
            "dataFormat_": data_format.name,
            "name_": name,
            "data_": {
                "dim_prop_func": [{"Const": {}}, {"Const": {}}, {"Map": {}}],
                "dim_prop_attr": [
                    {"factor_": num_cores, "label_": "core"},
                    {"factor_": 1, "label_": "corelet"},
                    {"factor_": 1, "label_": "time"},
                ],
                "data_": {"[0, 0, 0]": [encode_constant(value, data_format)]},
            },
        }
        constant_info[f"{len(constant_info)}"] = ci
    return constant_info


def add_constant(kwargs, name, value) -> int:
    """
    Add a constant to kwargs['op_info']['constants'] and return its index.
    Returns:
        int: The index of the newly added constant (0-based)
    """
    # Ensure structure exists
    if "op_info" not in kwargs:
        kwargs["op_info"] = {}
    if "constants" not in kwargs["op_info"]:
        kwargs["op_info"]["constants"] = {}

    index = len(kwargs["op_info"]["constants"])
    kwargs["op_info"]["constants"][name] = value

    return index


def gen_coord_info_value(
    size: int,
    nsplits: int,
    elems_per_stick: int,
    is_stick_dim: bool,
    is_stick_reduction: bool = False,
):
    return (
        {
            "spatial": 3,
            "temporal": 0,
            "elemArr": 1,
            "padding": "nopad",
            "folds": {
                "dim_prop_func": [
                    {
                        "Affine": {
                            "alpha_": size,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 1,
                            "beta_": 0,
                        }
                    },
                ],
                "dim_prop_attr": [
                    {
                        "factor_": nsplits,
                        "label_": "core_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "corelet_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "row_fold",
                    },
                    {
                        "factor_": size,
                        "label_": "elem_arr_0",
                    },
                ],
            },
        }
        if not is_stick_dim
        else {
            "spatial": 3,
            "temporal": 0,
            "elemArr": 2,
            "padding": "nopad",
            "folds": {
                "dim_prop_func": [
                    {
                        "Affine": {
                            "alpha_": elems_per_stick if is_stick_reduction else size,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": elems_per_stick,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0 if is_stick_reduction else 1,
                            "beta_": 0,
                        }
                    },
                ],
                "dim_prop_attr": [
                    {
                        "factor_": nsplits,
                        "label_": "core_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "corelet_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "row_fold",
                    },
                    {
                        "factor_": 1
                        if is_stick_reduction
                        else (size // elems_per_stick),
                        "label_": "elem_arr_1",
                    },
                    {
                        "factor_": elems_per_stick,
                        "label_": "elem_arr_0",
                    },
                ],
            },
        }
    )


def generate_sdsc(sdsc_spec):
    out_idx = len(sdsc_spec.args) - 1
    core_id_to_wk_slice = {
        str(c): {
            str(dim): int(expr.subs({Symbol("core_id"): c}))
            for dim, expr in sdsc_spec.core_id_to_work_slice.items()
        }
        for c in range(sdsc_spec.num_cores)
    }
    return {
        sdsc_spec.opfunc: {
            "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
            "sdscFolds_": {
                "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                "data_": {"[0]": "0"},
            },
            "coreFoldProp_": {"factor_": sdsc_spec.num_cores, "label_": "core"},
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numCoresUsed_": sdsc_spec.num_cores,
            "coreIdToDsc_": {str(c): 0 for c in range(sdsc_spec.num_cores)},
            "numWkSlicesPerDim_": {
                str(dim): num_wk_slices
                for dim, num_wk_slices in sdsc_spec.work_slices.items()
            },
            "coreIdToWkSlice_": core_id_to_wk_slice,
            "coreIdToDscSchedule": {
                str(c): [[-1, 0, 0, 0]] for c in range(sdsc_spec.num_cores)
            },
            "dscs_": [
                {
                    sdsc_spec.opfunc: {
                        "numCoresUsed_": sdsc_spec.num_cores,
                        "numCoreletsUsed_": 1,
                        "coreIdsUsed_": [c for c in range(sdsc_spec.num_cores)],
                        "N_": {
                            "name_": "n",
                            **{
                                str(dim) + "_": size
                                for dim, size in sdsc_spec.iteration_space.items()
                            },
                        },
                        "coordinateMasking_": {
                            str(dim): mask_range
                            for dim, mask_range in sdsc_spec.coordinate_masking.items()
                        },
                        "maskingConstId_": 0 if sdsc_spec.coordinate_masking else -1,
                        "dataStageParam_": {
                            "0": {
                                "ss_": {
                                    "name_": "core",
                                    **{
                                        str(dim) + "_": size
                                        // sdsc_spec.work_slices[dim]
                                        for dim, size in sdsc_spec.iteration_space.items()
                                    },
                                },
                                "el_": {
                                    "name_": "core",
                                    **{
                                        str(dim) + "_": size
                                        // sdsc_spec.work_slices[dim]
                                        for dim, size in sdsc_spec.iteration_space.items()
                                    },
                                },
                            }
                        },
                        "primaryDsInfo_": {
                            label: {
                                "layoutDimOrder_": [
                                    str(dim) for dim in layout_info["dim_order"]
                                ],
                                "stickDimOrder_": [str(layout_info["stick_dim_order"])],
                                "stickSize_": [layout_info["stick_size"]],
                            }
                            for label, layout_info in sdsc_spec.layouts.items()
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "name_": f"allocate-Tensor{i}_{'hbm' if not tensor.allocation else 'lx'}",
                                "prev_": "",
                                "ldsIdx_": i,
                                "component_": "hbm" if not tensor.allocation else "lx",
                                "layoutDimOrder_": [
                                    str(dim)
                                    for dim in sdsc_spec.layouts[tensor.layout][
                                        "dim_order"
                                    ]
                                ],
                                "maxDimSizes_": [
                                    tensor.max_dim_sizes[dim]
                                    for dim in sdsc_spec.layouts[tensor.layout][
                                        "dim_order"
                                    ]
                                ],
                                "startAddressCoreCorelet_": {
                                    "dim_prop_func": [
                                        {"Map": {}},
                                        {"Const": {}},
                                        {"Const": {}},
                                    ],
                                    "dim_prop_attr": [
                                        {
                                            "factor_": sdsc_spec.num_cores,
                                            "label_": "core",
                                        },
                                        {"factor_": 1, "label_": "corelet"},
                                        {"factor_": 1, "label_": "time"},
                                    ],
                                    "data_": {
                                        f"[{c}, 0, 0]": str(
                                            tensor.start_address
                                            + core_idx_to_slice_offset(
                                                tensor,
                                                core_id_to_wk_slice[str(c)],
                                                sdsc_spec.work_slices,
                                            )
                                            * num_bytes(tensor.data_format)
                                        )
                                        for c in range(sdsc_spec.num_cores)
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        str(dim): gen_coord_info_value(
                                            size=sdsc_spec.iteration_space[dim]
                                            // sdsc_spec.work_slices[dim]
                                            if (tensor.scales[dim] == 1)
                                            else 1,
                                            nsplits=sdsc_spec.work_slices[dim]
                                            if (tensor.scales[dim] == 1)
                                            else 1,
                                            elems_per_stick=tensor.data_format.elems_per_stick(),
                                            is_stick_dim=(
                                                sdsc_spec.layouts[tensor.layout][
                                                    "stick_dim_order"
                                                ].has(dim)
                                            ),
                                            is_stick_reduction=(
                                                tensor.scales[dim] == -2
                                            ),
                                        )
                                        for dim in sdsc_spec.layouts[tensor.layout][
                                            "dim_order"
                                        ]
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            }
                            for i, tensor in enumerate(sdsc_spec.args)
                        ],
                        "labeledDs_": [
                            {
                                "ldsIdx_": i,
                                "dsName_": f"Tensor{i}",
                                "dsType_": tensor.layout,
                                "scale_": [
                                    tensor.scales[dim]
                                    for dim in sdsc_spec.layouts[tensor.layout][
                                        "dim_order"
                                    ]
                                ],
                                "wordLength": num_bytes(tensor.data_format),
                                "dataFormat_": tensor.data_format.name,
                                "memOrg_": {
                                    "hbm": {"isPresent": 1},
                                    "lx": {"isPresent": 1},
                                }
                                if not tensor.allocation
                                else {"lx": {"isPresent": 1}},
                            }
                            for i, tensor in enumerate(sdsc_spec.args)
                        ],
                        "constantInfo_": generate_constant_info(
                            sdsc_spec.data_format,
                            sdsc_spec.constants,
                            sdsc_spec.num_cores,
                        ),
                        "computeOp_": [
                            {
                                "exUnit": sdsc_spec.execution_unit,
                                "opFuncName": sdsc_spec.opfunc,
                                "attributes_": {
                                    "dataFormat_": sdsc_spec.data_format.name,
                                    "fidelity_": "regular",
                                },
                                "location": "Inner",
                                "inputLabeledDs": [
                                    f"Tensor{i}-idx{i}"
                                    for i in range(sdsc_spec.num_inputs)
                                ],
                                "outputLabeledDs": [f"Tensor{out_idx}-idx{out_idx}"],
                            }
                        ],
                    }
                }
            ],
        }
    }


# Extract the padded sizes from the tensors for mm and bmm.
# The pattern of how op dimensions map to tensors seems to be:
#  - The first N-2 dims come from tensor 0
#  - Last 2 dims come from tensor 1
def get_padded_dimensions_matmul(ndim, inputs):
    padded_dimensions = [0] * ndim
    for op_dim in range(ndim):
        tensor_idx = 0 if op_dim < ndim - 2 else 1
        padded_dimensions[op_dim] = get_device_size(op_dim, inputs[tensor_idx])
    return padded_dimensions


def _generate_matmul_common(
    pointers,
    *,
    op,
    dimensions,
    inputs,
    outputs,
    dim_labels,
    dim_indices,
    dim_splits,
    cores,
):
    """
    Common implementation for matmul and bmm operations.

    This function contains the shared logic between generate_matmul and generate_bmm,
    which differ primarily in their dimension configurations.

    Args:
        pointers: Memory pointers for tensors
        op: Operation name
        dimensions: Tensor host dimensions
        inputs: Input tensor specifications
        outputs: Output tensor specifications
        dim_labels: Dimension labels (e.g., ["mb", "in", "out"] for matmul)
        dim_indices: Dimension indices
        dim_splits: Number of splits per dimension
        coreid_to_wk_slice: Mapping from core ID to work slice
        cores: Number of cores used

    Returns:
        Dictionary containing the SDSC structure for the operation
    """
    tensors = inputs + outputs
    data_format = inputs[0]["device_layout"].device_dtype

    padded_dimensions = get_padded_dimensions_matmul(len(dim_indices), inputs)

    dim_infos = DimInfos(
        dim_indices,
        dim_labels,
        dimensions,
        padded_dimensions,
        dim_splits,
    )

    layouts = create_tensor_specific_layouts(
        tensors, dim_infos, op, check_stick_dim=True
    )

    # swap_last_two_elements moves the "in" (reduction) dimension to the last
    # so that the core assignment keeps partial sum results that require cross-core
    # communications close
    coreid_to_wk_slice = calculate_core_to_slice_mapping(
        swap_last_two_elements(dim_labels),
        swap_last_two_elements(dim_splits),
    )

    return {
        op: {
            "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
            "sdscFolds_": {
                "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                "data_": {"[0]": "0"},
            },
            "coreFoldProp_": {"factor_": cores, "label_": "core"},
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numCoresUsed_": cores,
            "coreIdToDsc_": {str(i): 0 for i in range(cores)},
            "numWkSlicesPerDim_": {k: v for k, v in zip(dim_labels, dim_splits)},
            "coreIdToWkSlice_": coreid_to_wk_slice,
            "coreIdToDscSchedule": {str(i): [[-1, 0, 0, 0]] for i in range(cores)},
            "dscs_": [
                {
                    op: {
                        "numCoresUsed_": cores,
                        "numCoreletsUsed_": 1,
                        "coreIdsUsed_": [i for i in range(cores)],
                        "N_": {
                            "name_": "n",
                            **{
                                di.label + "_": di.padded_size
                                for di in dim_infos.get_op_infos()
                            },
                        },
                        "dataStageParam_": {
                            "0": {
                                "ss_": {
                                    "name_": "core",
                                    **{
                                        di.label + "_": di.split_size
                                        for di in dim_infos.get_op_infos()
                                    },
                                },
                                "el_": {
                                    "name_": "core",
                                    **{
                                        di.label + "_": di.split_size
                                        for di in dim_infos.get_op_infos()
                                    },
                                },
                            }
                        },
                        "primaryDsInfo_": {
                            name: {
                                "layoutDimOrder_": layout_info["layout_order"],
                                "stickDimOrder_": layout_info["stick_dim_order"],
                                "stickSize_": [data_format.elems_per_stick()],
                            }
                            for name, layout_info in layouts.items()
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                # "name_": node_name,
                                "name_": f"allocate_Input{idx}_hbm"
                                if idx < len(tensors) - 1
                                else "allocate_out_hbm",
                                "prev_": "",
                                "ldsIdx_": idx,
                                "component_": "hbm",
                                "layoutDimOrder_": dim_infos.get_tensor_layout_order(
                                    tensor
                                ),
                                "maxDimSizes_": [-1]
                                * len(dim_infos.get_tensor_layout_order(tensor)),
                                "startAddressCoreCorelet_": {
                                    "dim_prop_func": [
                                        {"Map": {}},
                                        {"Const": {}},
                                        {"Const": {}},
                                    ],
                                    "dim_prop_attr": [
                                        {"factor_": cores, "label_": "core"},
                                        {"factor_": 1, "label_": "corelet"},
                                        {"factor_": 1, "label_": "time"},
                                    ],
                                    "data_": {
                                        f"[{c}, 0, 0]": str(
                                            pointers[tensor["name"]]
                                            + core_idx_to_slice_offset(
                                                dim_infos.get_tensor_infos(tensor, op),
                                                coreid_to_wk_slice[str(c)],
                                                tensor["device_layout"].device_size,
                                            )
                                            * num_bytes(
                                                tensor["device_layout"].device_dtype
                                            )
                                        )
                                        for c in range(cores)
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        di.label: gen_coord_info_value(
                                            size=di.split_size
                                            if (di.scale == 1)
                                            else 1,
                                            nsplits=di.nsplits,
                                            elems_per_stick=tensor[
                                                "device_layout"
                                            ].device_dtype.elems_per_stick(),
                                            is_stick_dim=(
                                                di.label
                                                in dim_infos.get_tensor_stick_dim_labels(
                                                    tensor
                                                )
                                            ),
                                        )
                                        for di in dim_infos.get_tensor_infos(tensor, op)
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            }
                            for idx, tensor in enumerate(tensors)
                        ],
                        "labeledDs_": [
                            {
                                "ldsIdx_": idx,
                                "dsName_": f"Tensor{idx}",
                                "dsType_": tensor["ds_type"],
                                "scale_": [
                                    di.scale
                                    for di in dim_infos.get_tensor_infos(tensor, op)
                                ],
                                "wordLength": num_bytes(
                                    tensor["device_layout"].device_dtype
                                ),
                                "dataFormat_": tensor[
                                    "device_layout"
                                ].device_dtype.name,
                                "memOrg_": {
                                    "hbm": {"isPresent": 1},
                                    "lx": {"isPresent": 1},
                                },
                            }
                            for idx, tensor in enumerate(tensors)
                        ],
                        "computeOp_": [
                            {
                                "exUnit": "pt",
                                "opFuncName": op,
                                "attributes_": {
                                    "dataFormat_": inputs[0][
                                        "device_layout"
                                    ].device_dtype.name,
                                    "fidelity_": "regular",
                                },
                                "location": "Inner",
                                "inputLabeledDs": [
                                    "Tensor0-idx0",
                                    "Tensor1-idx1",
                                ],
                                "outputLabeledDs": ["Tensor2-idx2"],
                            }
                        ],
                    }
                }
            ],
        }
    }


def generate_matmul(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    """
    Generate SDSC structure for matrix multiplication operation.

    Matmul operation: [mb=dim0, in=dim1] @ [in=dim1, out=dim2]

    This is a thin wrapper around _generate_matmul_common that provides
    matmul-specific configuration (3D dimensions, specific layouts).
    """
    dim_labels = ["mb", "in", "out"]
    dim_indices = [0, 1, 2]

    # work division logic
    cores = 1
    dim_splits = [1, 1, 1]
    if "op_info" in kwargs:
        if "n_cores_used" in kwargs["op_info"]:
            cores = kwargs["op_info"]["n_cores_used"]

        if "op_dim_splits" in kwargs["op_info"]:
            dim_splits = list(kwargs["op_info"]["op_dim_splits"])

    return _generate_matmul_common(
        pointers,
        op=op,
        dimensions=dimensions,
        inputs=inputs,
        outputs=outputs,
        dim_labels=dim_labels,
        dim_indices=dim_indices,
        dim_splits=dim_splits,
        cores=cores,
    )


def generate_bmm(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    """
    Generate SDSC structure for batched matrix multiplication operation.

    BMM operation: [x=dim0, mb=dim1, in=dim2] @ [x=dim0, in=dim2, out=dim3]

    This is a thin wrapper around _generate_matmul_common that provides
    bmm-specific configuration (4D dimensions with batch, specific layouts).
    """
    if len(dimensions) == 4:  # 3d bmm
        dim_labels = ["x", "mb", "in", "out"]
    else:  # 4d bmm
        dim_labels = ["x", "y", "mb", "in", "out"]

    dim_indices = list(range(len(dim_labels)))

    cores = 1
    dim_splits = [1] * len(dim_labels)
    if "op_info" in kwargs:
        if "n_cores_used" in kwargs["op_info"]:
            cores = kwargs["op_info"]["n_cores_used"]

        if "op_dim_splits" in kwargs["op_info"]:
            dim_splits = list(kwargs["op_info"]["op_dim_splits"])

    return _generate_matmul_common(
        pointers,
        op=op,
        dimensions=dimensions,
        inputs=inputs,
        outputs=outputs,
        dim_labels=dim_labels,
        dim_indices=dim_indices,
        dim_splits=dim_splits,
        cores=cores,
    )


def generate_fused_bmm_softmax(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    """
    Generate SuperDSC for fused BMM + Softmax attention kernel.

    This creates a fused kernel that combines:
    1. Batch matrix multiplication: scores = Q @ K^T
    2. Softmax normalization: attn_weights = softmax(scores)

    The fusion enables:
    - Tiled computation for memory efficiency
    - Reduced memory bandwidth (no intermediate materialization)
    - Flash attention style optimization

    Args:
        pointers: Memory segment offsets
        op: Operation name ("fused_bmm_softmax")
        dimensions: Output dimensions [B, H, S_q, S_k]
        inputs: List of input tensor descriptors (query, key_transposed)
        outputs: List of output tensor descriptors
        **kwargs: Additional arguments including op_info with scaling_factor

    Returns:
        SuperDSC dictionary for the fused attention kernel
    """
    from .compute_ops import (
        DimInfos,
        create_tensor_specific_layouts,
        calculate_core_to_slice_mapping,
        generate_constant_info,
    )

    tensors = inputs + outputs
    data_format = tensors[0]["device_layout"].device_dtype
    ndim = len(dimensions)

    # Extract configuration
    batch, heads, seq_q, seq_k = dimensions
    op_info = kwargs.get("op_info", {})
    scaling_factor = op_info.get("constants", {}).get("scaling_factor", 1.0)

    # Determine tile sizes (optimize based on LX memory capacity)
    # These can be tuned for specific hardware configurations
    Q_TILE = 8
    K_TILE = 8
    SEQ_TILE = 8

    # Core configuration
    cores = kwargs.get("cores", [1, 1, 1, 1])
    dim_splits = [
        max(1, batch // cores[0]),
        max(1, heads // cores[1]),
        max(1, seq_q // SEQ_TILE),
        max(1, seq_k // SEQ_TILE),
    ]

    # Dimension labels for tiling
    dim_labels = ["batch", "heads", "seq_q", "seq_k"]
    dim_indices = list(range(ndim))

    # Padded dimensions for alignment
    padded_dimensions = [
        batch,
        heads,
        ((seq_q + SEQ_TILE - 1) // SEQ_TILE) * SEQ_TILE,
        ((seq_k + SEQ_TILE - 1) // SEQ_TILE) * SEQ_TILE,
    ]

    # Create dimension info structure
    dim_infos = DimInfos(
        dim_indices=dim_indices,
        labels=dim_labels,
        unpadded_sizes=dimensions,
        padded_sizes=padded_dimensions,
        nsplits=dim_splits,
    )

    # Generate tensor layouts
    layouts = create_tensor_specific_layouts(
        tensors=tensors,
        dim_infos=dim_infos,
        op=op,
        is_matmul=True,
        op_dims_tensor=inputs[0],
    )

    # Calculate core-to-slice mapping
    core_id_to_wk_slice = calculate_core_to_slice_mapping(
        dim_labels=dim_labels,
        dim_splits=dim_splits,
    )

    # Generate constant information
    constant_info = generate_constant_info(
        data_format=data_format,
        scaling_factor=scaling_factor,
    )

    # Build the SuperDSC structure
    superdsc = {
        "fused_attention_bmm_softmax": {
            "sdscFoldProps_": [
                {"size": Q_TILE, "label": "Q_tile"},
                {"size": K_TILE, "label": "K_tile"},
                {"size": SEQ_TILE, "label": "seq_tile"},
            ],
            "numCoresUsed_": 1,
            "dscs_": [
                {
                    "name_": "fused_bmm_softmax_dsc",
                    "computeOp_": "FUSED_BMM_SOFTMAX",
                    "dataflow_": "KG3_BMM_INT8",
                    "fusedOps_": [
                        {
                            "opType": "BatchMatMulV2",
                            "opFunc": "MACC",
                            "inputs": ["Q_tiled", "K_tiled"],
                            "output": "attention_scores_partial",
                        },
                        {
                            "opType": "Softmax",
                            "opFunc": "EXP",
                            "inputs": ["attention_scores_partial"],
                            "output": "attention_weights_partial",
                            "axis": -1,
                            "partialCompute": True,
                        },
                    ],
                    "memoryStrategy_": {
                        "LX": {
                            "stationary": ["K_tiled", "partial_max", "partial_sum"],
                            "streaming": ["Q_tiled"],
                            "accumulate": ["attention_scores_partial"],
                        },
                        "computation": {
                            "phase1": "BMM_compute_partial_scores",
                            "phase2": "Softmax_partial_normalization",
                        },
                    },
                    "tiling_": {
                        "Q_dim": Q_TILE,
                        "K_dim": K_TILE,
                        "seq_dim": SEQ_TILE,
                        "strategy": "flash_attention_style",
                    },
                    "layouts_": layouts,
                    "constantInfo_": constant_info,
                }
            ],
            "dataOpdscs_": [
                {
                    "name_": "load_Q_tile",
                    "transferType": "HBM_to_LX",
                    "source": "Q_cache",
                    "destination": "LX_Q_buffer",
                    "tileSize": [Q_TILE, K_TILE],
                },
                {
                    "name_": "load_K_tile",
                    "transferType": "HBM_to_LX",
                    "source": "K_cache",
                    "destination": "LX_K_buffer",
                    "tileSize": [K_TILE, SEQ_TILE],
                    "reuse": True,
                },
                {
                    "name_": "partial_result_accumulate",
                    "transferType": "LX_internal",
                    "operation": "accumulate_softmax_partial",
                    "keepInLX": True,
                },
            ],
            "coreIdToDscSchedule": core_id_to_wk_slice,
        }
    }

    return superdsc
