# Copyright 2023 The Kubeflow Authors. All Rights Reserved.
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

from typing import Dict, List

from google_cloud_pipeline_components.types.artifact_types import BQMLModel
from kfp.dsl import Artifact
from kfp.dsl import ConcatPlaceholder
from kfp.dsl import container_component
from kfp.dsl import ContainerSpec
from kfp.dsl import Input
from kfp.dsl import Output
from kfp.dsl import OutputPath


@container_component
def bigquery_ml_arima_coefficients(
    project: str,
    model: Input[BQMLModel],
    arima_coefficients: Output[Artifact],
    gcp_resources: OutputPath(str),
    location: str = 'us-central1',
    query_parameters: List[str] = [],
    job_configuration_query: Dict[str, str] = {},
    labels: Dict[str, str] = {},
    encryption_spec_key_name: str = '',
):
  """Launch a BigQuery ML.ARIMA_COEFFICIENTS job and let you see the ARIMA coefficients.

  This function only applies to the time-series ARIMA_PLUS and ARIMA models.

    Args:
        project (str):
          Required. Project to run the BigQuery job.
        location (Optional[str]):
          Location to run the BigQuery job. If not set,
          default to `US` multi-region. For more details, see
          https://cloud.google.com/bigquery/docs/locations#specifying_your_location
        model (google.BQMLModel):
          Required. BigQuery ML model for
          ML.ARIMA_COEFFICIENTS. For more details, see
          https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-arima-coefficients
        query_parameters (Optional[Sequence]):
          Query parameters for
          standard SQL queries. If query_parameters are both specified in here
          and in job_configuration_query, the value in here will override the
          other one.
        job_configuration_query (Optional[dict]):
          A json formatted string
          describing the rest of the job configuration. For more details, see
          https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationQuery
        encryption_spec_key_name(Optional[List[str]]):
          Describes the Cloud
          KMS encryption key that will be used to protect destination
          BigQuery table. The BigQuery Service Account associated with your
          project requires access to this encryption key. If
          encryption_spec_key_name are both specified in here and in
          job_configuration_query, the value in here will override the other
          one.

    Returns:
        arima_coefficients (system.Artifact):
            Describes arima_coefficients to the type of model supplied.
            For more details, see
            https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-arima-coefficients#mlarima_coefficients_output
        gcp_resources (str):
            Serialized gcp_resources proto tracking the BigQuery job.
            For more details, see
            https://github.com/kubeflow/pipelines/blob/master/components/google-cloud/google_cloud_pipeline_components/proto/README.md.
  """
  return ContainerSpec(
      image='gcr.io/ml-pipeline/google-cloud-pipeline-components:2.0.0b1',
      command=[
          'python3',
          '-u',
          '-m',
          'google_cloud_pipeline_components.container.v1.bigquery.ml_arima_coefficients.launcher',
      ],
      args=[
          '--type',
          'BigqueryMLArimaCoefficientsJob',
          '--project',
          project,
          '--location',
          location,
          '--model_name',
          ConcatPlaceholder([
              "{{$.inputs.artifacts['model'].metadata['projectId']}}",
              '.',
              "{{$.inputs.artifacts['model'].metadata['datasetId']}}",
              '.',
              "{{$.inputs.artifacts['model'].metadata['modelId']}}",
          ]),
          '--payload',
          ConcatPlaceholder([
              '{',
              '"configuration": {',
              '"query": ',
              job_configuration_query,
              ', "labels": ',
              labels,
              '}',
              '}',
          ]),
          '--job_configuration_query_override',
          ConcatPlaceholder([
              '{',
              '"query_parameters": ',
              query_parameters,
              ', "destination_encryption_configuration": {',
              '"kmsKeyName": "',
              encryption_spec_key_name,
              '"}',
              '}',
          ]),
          '--gcp_resources',
          gcp_resources,
          '--executor_input',
          '{{$}}',
      ],
  )
