nextflow.enable.dsl = 2

include { BASICPY } from './modules/local/basicpy'
include { ECC_RPCA } from './modules/local/ecc_rpca'
include { CELLPOSE } from './modules/local/cellpose'
include { ULTRACK } from './modules/local/ultrack'
include { QC } from './modules/local/qc'
include { PROVENANCE } from './modules/local/provenance'


def sourceHash(List paths) {
    def digest = java.security.MessageDigest.getInstance('SHA-256')
    paths.each { path -> digest.update(file(path).bytes) }
    return digest.digest().encodeHex().toString()
}


def sampleId(path) {
    def name = path.getName().replaceFirst(/(?i)\.(tif|tiff)$/, '')
    return name.replaceAll(/[^A-Za-z0-9._-]/, '_')
}


def frameList(value) {
    if (value == null) {
        return []
    }
    if (value instanceof List) {
        return value.collect { it as Integer }
    }
    def text = value.toString().trim()
    if (!text) {
        return []
    }
    return text.split(/[,\s]+/).findAll { it }.collect { it as Integer }
}


workflow {
    if (!params.input) {
        error "Missing required parameter --input (a TIFF path or glob)"
    }

    raw_movies = Channel
        .fromPath(params.input, checkIfExists: true)
        .ifEmpty { error "No input TIFF files matched: ${params.input}" }
        .map { movie -> tuple(sampleId(movie), movie) }
        .groupTuple()
        .map { id, movies ->
            if (movies.size() != 1) {
                error "Input files must have unique TIFF basenames; sample ID '${id}' matched ${movies.size()} files"
            }
            tuple(id, movies.first())
        }

    basicpy_hash = sourceHash([
        "${projectDir}/src/cell_track/common.py",
        "${projectDir}/src/cell_track/basicpy.py",
        "${projectDir}/src/cell_track/cli/basicpy.py",
    ])
    ecc_rpca_hash = sourceHash([
        "${projectDir}/src/cell_track/common.py",
        "${projectDir}/src/cell_track/ecc_rpca.py",
        "${projectDir}/src/cell_track/cli/ecc_rpca.py",
    ])
    cellpose_hash = sourceHash([
        "${projectDir}/src/cell_track/common.py",
        "${projectDir}/src/cell_track/segmentation.py",
        "${projectDir}/src/cell_track/cli/cellpose.py",
    ])
    ultrack_hash = sourceHash([
        "${projectDir}/src/cell_track/common.py",
        "${projectDir}/src/cell_track/tracking.py",
        "${projectDir}/src/cell_track/cli/ultrack.py",
    ])
    qc_hash = sourceHash([
        "${projectDir}/src/cell_track/common.py",
        "${projectDir}/src/cell_track/qc.py",
        "${projectDir}/src/cell_track/cli/qc.py",
    ])
    pipeline_hash = sourceHash([
        "${projectDir}/main.nf",
        "${projectDir}/nextflow.config",
        "${projectDir}/modules/local/basicpy.nf",
        "${projectDir}/modules/local/ecc_rpca.nf",
        "${projectDir}/modules/local/cellpose.nf",
        "${projectDir}/modules/local/ultrack.nf",
        "${projectDir}/modules/local/qc.nf",
        "${projectDir}/modules/local/provenance.nf",
    ])
    provenance_json = groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson([
            parameters: params,
            pipeline_source_hash: pipeline_hash,
        ])
    )

    PROVENANCE(provenance_json)

    BASICPY(
        raw_movies,
        params.basicpy_align_frames,
        params.basicpy_upsample_factor,
        params.basicpy_get_darkfield,
        params.basicpy_autotune,
        params.save_aligned_stacks,
        basicpy_hash,
    )

    ECC_RPCA(
        BASICPY.out.corrected,
        params.ecc_motion,
        params.ecc_max_iterations,
        params.ecc_epsilon,
        params.ecc_gaussian_filter_size,
        params.ecc_background_percentile,
        params.rpca_lambda_multiplier,
        params.rpca_max_iterations,
        params.rpca_tolerance,
        params.save_aligned_stacks,
        params.save_intermediates,
        ecc_rpca_hash,
    )

    CELLPOSE(
        ECC_RPCA.out.foreground,
        params.cellpose_model,
        params.cellpose_fallback_model,
        params.cellpose_device,
        params.cellpose_diameter == null ? '' : params.cellpose_diameter,
        params.cellpose_flow_threshold,
        params.cellpose_cellprob_threshold,
        params.cellpose_batch_size,
        params.cellpose_niter == null ? '' : params.cellpose_niter,
        params.cellpose_bsize == null ? '' : params.cellpose_bsize,
        params.cellpose_gamma,
        params.cellpose_lower_quantile,
        params.cellpose_upper_quantile,
        params.save_normalized_stack,
        cellpose_hash,
    )

    ULTRACK(
        CELLPOSE.out.labels,
        params.ultrack_contour_sigma,
        params.ultrack_min_area,
        params.ultrack_max_area,
        params.ultrack_max_distance,
        params.ultrack_n_workers,
        params.ultrack_appear_weight,
        params.ultrack_disappear_weight,
        params.ultrack_division_weight,
        params.ultrack_power,
        params.ultrack_bias,
        params.ultrack_solution_gap,
        params.ultrack_time_limit,
        params.ultrack_solver_name,
        params.save_ultrack_database,
        ultrack_hash,
    )

    qc_inputs = ECC_RPCA.out.foreground
        .join(CELLPOSE.out.labels, by: 0)
        .join(ULTRACK.out.results, by: 0)

    QC(qc_inputs, frameList(params.qc_frames), qc_hash)
}
