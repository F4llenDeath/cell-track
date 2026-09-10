process ECC_RPCA {
    tag "$sample_id"
    label 'ecc_rpca'

    publishDir { "${params.outdir}/${sample_id}/preprocessing" }, mode: 'copy', overwrite: true,
        saveAs: { name ->
            if (name == 'ecc_rpca_diagnostics') {
                return 'ecc_rpca_diagnostics'
            }
            return params.save_intermediates ? name : null
        }

    input:
    tuple val(sample_id), path(corrected_tif)
    val motion
    val ecc_max_iterations
    val ecc_epsilon
    val gaussian_filter_size
    val background_percentile
    val lambda_multiplier
    val rpca_max_iterations
    val rpca_tolerance
    val save_aligned_stack
    val save_ecc_residual
    val code_hash

    output:
    tuple val(sample_id), path("${sample_id}_rpca_corrected.tif"), emit: foreground
    tuple val(sample_id), path('ecc_rpca_diagnostics'), emit: diagnostics
    tuple val(sample_id), path("${sample_id}_basicpy_ecc_corrected.tif"), optional: true, emit: ecc_residual

    script:
    def aligned_arg = save_aligned_stack ? '--save-aligned-stack' : '--no-save-aligned-stack'
    def residual_arg = save_ecc_residual ? '--save-ecc-residual' : '--no-save-ecc-residual'
    """
    # cell-track source hash: ${code_hash}
    export PYTHONPATH="${projectDir}/src:\${PYTHONPATH:-}"
    python -m cell_track.cli.ecc_rpca \
        --input "${corrected_tif}" \
        --output-dir . \
        --prefix "${sample_id}" \
        --motion ${motion} \
        --ecc-max-iterations ${ecc_max_iterations} \
        --ecc-epsilon ${ecc_epsilon} \
        --gaussian-filter-size ${gaussian_filter_size} \
        --background-percentile ${background_percentile} \
        --lambda-multiplier ${lambda_multiplier} \
        --rpca-max-iterations ${rpca_max_iterations} \
        --rpca-tolerance ${rpca_tolerance} \
        ${aligned_arg} \
        ${residual_arg} \
        --progress
    """

    stub:
    """
    mkdir -p ecc_rpca_diagnostics
    touch "${sample_id}_rpca_corrected.tif"
    printf '{"stub": true}\n' > ecc_rpca_diagnostics/metadata.json
    if [[ "${save_ecc_residual}" == "true" ]]; then
        touch "${sample_id}_basicpy_ecc_corrected.tif"
    fi
    """
}
