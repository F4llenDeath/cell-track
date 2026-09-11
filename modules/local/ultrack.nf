process ULTRACK {
    tag "$sample_id"
    label 'ultrack'

    publishDir { "${params.outdir}/${sample_id}/tracking" }, mode: 'copy', overwrite: true

    input:
    tuple val(sample_id), path(cellpose_labels)
    val contour_sigma
    val min_area
    val max_area
    val max_distance
    val n_workers
    val appear_weight
    val disappear_weight
    val division_weight
    val power
    val bias
    val solution_gap
    val time_limit
    val solver_name
    val save_database
    val code_hash

    output:
    tuple val(sample_id), path('tracks_df.csv'), path('tracked_labels.tif'), emit: results
    tuple val(sample_id), path('cell_areas.csv'), path('tracking_metadata.json'), emit: reports
    tuple val(sample_id), path('ultrack_databases'), optional: true, emit: database

    script:
    def database_arg = save_database ? '--save-database' : '--no-save-database'
    def effective_workers = Math.max(1, Math.min(n_workers as Integer, task.cpus as Integer))
    """
    # cell-track source hash: ${code_hash}
    export PYTHONPATH="${projectDir}/src:\${PYTHONPATH:-}"
    python -m cell_track.cli.ultrack \
        --input "${cellpose_labels}" \
        --output-dir . \
        --working-dir ultrack_work \
        --contour-sigma ${contour_sigma} \
        --min-area ${min_area} \
        --max-area ${max_area} \
        --max-distance ${max_distance} \
        --n-workers ${effective_workers} \
        --appear-weight ${appear_weight} \
        --disappear-weight ${disappear_weight} \
        --division-weight ${division_weight} \
        --power ${power} \
        --bias ${bias} \
        --solution-gap ${solution_gap} \
        --time-limit ${time_limit} \
        --solver-name "${solver_name}" \
        ${database_arg}
    """

    stub:
    """
    printf 'track_id,t,y,x,id,parent_track_id,parent_id\n' > tracks_df.csv
    touch tracked_labels.tif
    printf 'area\n' > cell_areas.csv
    printf '{"stub": true}\n' > tracking_metadata.json
    if [[ "${save_database}" == "true" ]]; then
        mkdir -p ultrack_databases
        printf '{"segments": []}\n' > ultrack_databases/manifest.json
    fi
    """
}
