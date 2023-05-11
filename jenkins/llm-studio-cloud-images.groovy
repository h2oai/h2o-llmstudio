import org.jenkinsci.plugins.pipeline.modeldefinition.Utils

properties(
    [
        parameters(
            [
                string(name: 'BRANCH_TAG', defaultValue: 'origin/main'),
                booleanParam(name: 'AWS', defaultValue: true, description: 'Make Amazon Machine Image/Not?'),
                booleanParam(name: 'GCP', defaultValue: true, description: 'Make GCP Image/Not?'),
                booleanParam(name: 'AZURE', defaultValue: true, description: 'Make AZURE Image/Not?'),
                string(name: 'LLM_STUDIO_VERSION')
            ]
        )
    ]
)

node('mr-0x16') {
    stage('Init') {
        cleanWs()
        currentBuild.displayName = "#${BUILD_NUMBER} - Rel:${LLM_STUDIO_VERSION}"
        checkout scm
        sh('ls -al')
    }

    stage('Build Images') {
        try {
            docker.image('harbor.h2o.ai/opsh2oai/h2oai-packer-build:2').inside {
                parallel([
                        "AWS Ubuntu 20.04": {
                            withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'jenkins-full-aws-creds'],
                                             string(credentialsId: "AWS_MARKETPLACE_VPC", variable: "aws_vpc_id"),
                                             string(credentialsId: "AWS_MARKETPLACE_OWNERS", variable: "aws_owners"),
                                             string(credentialsId: "AWS_MARKETPLACE_SUBNET", variable: "aws_subnet_id"),
                                             string(credentialsId: "AWS_MARKETPLACE_SG", variable: "aws_security_group_id")]) {
                                dir('jenkins') {
                                    if (params.AWS) {
                                        sh("packer build \
                                        -var 'aws_access_key=$AWS_ACCESS_KEY_ID' \
                                        -var 'aws_secret_key=$AWS_SECRET_ACCESS_KEY' \
                                        -var 'llm_studio_version=${LLM_STUDIO_VERSION}' \
                                        -var 'aws_region=us-east-1' \
                                        -var 'aws_vpc_id=$aws_vpc_id' \
                                        -var 'aws_owners=$aws_owners' \
                                        -var 'aws_subnet_id=$aws_subnet_id' \
                                        -var 'aws_security_group_id=$aws_security_group_id' \
                                        llm-studio-aws.json"
                                        )
                                        archiveArtifacts artifacts: '*-image-info.json'
                                    }else {
                                        Utils.markStageSkippedForConditional('AWS Ubuntu 20.04')
                                    }
                                }
                            }
                        },

                        "GCP Ubuntu 20.04": {
                            withCredentials([file(credentialsId: 'GCP_MARKETPLACE_SERVICE_ACCOUNT', variable: 'GCP_ACCOUNT_FILE')]) {
                                dir('jenkins') {
                                    if (params.GCP) {
                                        sh("packer build \
                                            -var 'project_id=h2o-gce' \
                                            -var 'account_file=$GCP_ACCOUNT_FILE' \
                                            -var 'llm_studio_version=${LLM_STUDIO_VERSION}' \
                                            llm-studio-gcp.json"
                                        )
                                        archiveArtifacts artifacts: '*-image-info.json'
                                    }else {
                                        Utils.markStageSkippedForConditional('GCP Ubuntu 20.04')
                                    }
                                }
                            }
                        },

                         "AZURE Ubuntu 20.04": {
                            withCredentials([string(credentialsId: "AZURE_MARKETPLACE_CLIENT_ID", variable: "AZURE_CLIENT_ID"),
                                             string(credentialsId: "AZURE_MARKETPLACE_CLIENT_SECRET", variable: "AZURE_CLIENT_SECRET"),
                                             string(credentialsId: "AZURE_MARKETPLACE_SUBSCRIPTION_ID", variable: "AZURE_SUBSCRIPTION_ID"),
                                             string(credentialsId: "AZURE_MARKETPLACE_TENANT_ID", variable: "AZURE_TENANT_ID")]) {
                                dir('jenkins') {
                                    if (params.AZURE) {
                                        sh("packer build \
                                            -var 'client_id=$AZURE_CLIENT_ID' \
                                            -var 'client_secret=$AZURE_CLIENT_SECRET' \
                                            -var 'managed_image_resource_group_name=H2OIMAGES' \
                                            -var 'subscription_id=$AZURE_SUBSCRIPTION_ID' \
                                            -var 'tenant_id=$AZURE_TENANT_ID' \
                                            -var 'llm_studio_version=${LLM_STUDIO_VERSION}' \
                                            llm-studio-azure.json"
                                        )
                                        archiveArtifacts artifacts: '*-image-info.json'
                                    }else {
                                        Utils.markStageSkippedForConditional('AZURE Ubuntu 20.04')
                                    }
                                }
                            }
                        },

                ])
            }
        } finally {
            cleanWs()
        }
    }
}