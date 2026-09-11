process CELLPOSE {
    tag "$sample_id"
    label 'cellpose'

    publishDir { "${params.outdir}/${sample_id}/segmentation" }, mode: 'copy', overwrite: true

    input:
    tuple val(sample_id), path(foreground_tif)
    val model
    val fallback_model
    val device
    val diameter
    val flow_threshold
    val cellprob_threshold
    val batch_size
    val niter
    val bsize
    val gamma
    val lower_quantile
    val upper_quantile
    val save_normalized_stack
    val code_hash

    output:
    tuple val(sample_id), path('cellpose_labels.tif'), emit: labels
    tuple val(sample_id), path('cell_counts.csv'), path('cellpose_metadata.json'), emit: reports
    tuple val(sample_id), path('normalized_foreground.tif'), optional: true, emit: normalized

    script:
    def diameter_arg = diameter == '' ? '' : "--diameter ${diameter}"
    def niter_arg = niter == '' ? '' : "--niter ${niter}"
    def bsize_arg = bsize == '' ? '' : "--bsize ${bsize}"
    def normalized_arg = save_normalized_stack ? '--save-normalized-stack' : '--no-save-normalized-stack'
    """
    # cell-track source hash: ${code_hash}
    export PYTHONPATH="${projectDir}/src:\${PYTHONPATH:-}"
    python -m cell_track.cli.cellpose \
        --input "${foreground_tif}" \
        --output-dir . \
        --model "${model}" \
        --fallback-model "${fallback_model}" \
        --device ${device} ${diameter_arg} \
        --flow-threshold ${flow_threshold} \
        --cellprob-threshold ${cellprob_threshold} \
        --batch-size ${batch_size} ${niter_arg} ${bsize_arg} \
        --gamma ${gamma} \
        --lower-quantile ${lower_quantile} \
        --upper-quantile ${upper_quantile} \
        ${normalized_arg} \
        --progress
    """

    stub:
    """
    touch cellpose_labels.tif
    printf 't,cell_count\n' > cell_counts.csv
    printf '{"stub": true}\n' > cellpose_metadata.json
    if [[ "${save_normalized_stack}" == "true" ]]; then
        touch normalized_foreground.tif
    fi
    """
}
