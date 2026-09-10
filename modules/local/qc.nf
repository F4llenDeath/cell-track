process QC {
    tag "$sample_id"
    label 'qc'

    publishDir { "${params.outdir}/${sample_id}/qc" }, mode: 'copy', overwrite: true

    input:
    tuple val(sample_id), path(foreground_tif), path(cellpose_labels), path(tracks_csv), path(tracked_labels)
    val qc_frames
    val code_hash

    output:
    tuple val(sample_id), path('cells_per_frame.png'), path('segmentation_overlays.png'), path('track_durations.png'), path('tracking_overlays.png'), path('summary.json'), emit: reports

    script:
    def frames_arg = qc_frames ? "--frames ${qc_frames.join(' ')}" : ''
    """
    # cell-track source hash: ${code_hash}
    export PYTHONPATH="${projectDir}/src:\${PYTHONPATH:-}"
    python -m cell_track.cli.qc \
        --foreground "${foreground_tif}" \
        --cellpose-labels "${cellpose_labels}" \
        --tracks "${tracks_csv}" \
        --tracked-labels "${tracked_labels}" \
        --output-dir . ${frames_arg}
    """

    stub:
    """
    touch cells_per_frame.png segmentation_overlays.png track_durations.png tracking_overlays.png
    printf '{"stub": true}\n' > summary.json
    """
}
