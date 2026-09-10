process PROVENANCE {
    label 'provenance'

    publishDir { "${params.outdir}/pipeline_info" }, mode: 'copy', overwrite: true

    input:
    val provenance_json

    output:
    path 'parameters.json'

    script:
    def escaped_json = provenance_json.replace("'", "'\"'\"'")
    """
    printf '%s\n' '${escaped_json}' > parameters.json
    """

}
