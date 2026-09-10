process BASICPY {
    tag "$sample_id"
    label 'basicpy'

    publishDir { "${params.outdir}/${sample_id}/preprocessing" }, mode: 'copy', overwrite: true,
        saveAs: { name ->
            if (name == 'basicpy_diagnostics') {
                return 'basicpy_diagnostics'
            }
            return params.save_intermediates ? name : null
        }

    input:
    tuple val(sample_id), path(raw_tif)
    val align_frames
    val upsample_factor
    val get_darkfield
    val autotune
    val save_aligned_stack
    val code_hash

    output:
    tuple val(sample_id), path("${sample_id}_basicpy_timelapse.tif"), emit: corrected
    tuple val(sample_id), path("${sample_id}_basicpy_flatfield_only.tif"), path('basicpy_diagnostics'), emit: artifacts

    script:
    def align_arg = align_frames ? '--align-frames' : '--no-align-frames'
    def darkfield_arg = get_darkfield ? '--get-darkfield' : '--no-get-darkfield'
    def autotune_arg = autotune ? '--autotune' : '--no-autotune'
    def aligned_arg = save_aligned_stack ? '--save-aligned-stack' : '--no-save-aligned-stack'
    """
    # cell-track source hash: ${code_hash}
    export PYTHONPATH="${projectDir}/src:\${PYTHONPATH:-}"
    python -m cell_track.cli.basicpy \
        --input "${raw_tif}" \
        --output-dir . \
        --prefix "${sample_id}" \
        ${align_arg} \
        --upsample-factor ${upsample_factor} \
        ${darkfield_arg} \
        ${autotune_arg} \
        ${aligned_arg} \
        --progress
    """

    stub:
    """
    mkdir -p basicpy_diagnostics
    touch "${sample_id}_basicpy_timelapse.tif"
    touch "${sample_id}_basicpy_flatfield_only.tif"
    printf '{"stub": true}\n' > basicpy_diagnostics/metadata.json
    """
}
