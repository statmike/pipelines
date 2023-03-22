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

import json
import logging
import re

import google.auth
from google_cloud_pipeline_components.proto.gcp_resources_pb2 import GcpResources
import googleapiclient.discovery as discovery
import requests

from google.protobuf.json_format import Parse

_SUPPORTED_JOB_TYPES = ['DataflowJob',
                        'VertexLro',
                        'BigQueryJob',
                        'BatchPredictionJob',
                        'HyperparameterTuningJob',
                        'CustomJob',
                        'DataprocLro']
_JOB_CANCELLED_STATE = 'JOB_STATE_CANCELLED'
_DATAFLOW_URI_TEMPLATE = r'(https://dataflow.googleapis.com/v1b3/projects/(?P<project>.*)/locations/(?P<location>.*)/jobs/(?P<jobid>.*))'
_JOB_URI_TEMPLATE = r'https://(?P<location>.*)-aiplatform.googleapis.com/v1/(?P<jobname>.*)'


def cancel_job(
    gcp_resources,
):
  """Cancels a running job."""
  if gcp_resources == 'default':
    raise ValueError(
        'No gcp_resources provided, Job may have already completed'
    )

  gcp_resources = Parse(gcp_resources, GcpResources())
  resource_type = gcp_resources.resources[0].resource_type
  job_uri = gcp_resources.resources[0].resource_uri

  if len(gcp_resources.resources) != 1:
    raise ValueError(
        'Invalid gcp_resources: %s. Cancel job component supports cancelling'
        ' only one resource at this moment.' % gcp_resources
    )
  if resource_type not in _SUPPORTED_JOB_TYPES:
    raise ValueError(
        'Invalid gcp_resources: %s. Resource type not supported' % gcp_resources
    )

  def cancel_job_using_uri(job_uri, resource_type):
    creds, _ = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )

    if not creds.valid:
      creds.refresh(google.auth.transport.requests.Request())
    headers = {
        'Content-type': 'application/json',
        'Authorization': 'Bearer ' + creds.token,
    }
    if resource_type in [
        'BatchPredictionJob',
        'HyperparameterTuningJob',
        'CustomJob',
    ]:
      response = requests.post(
          url=f'{job_uri}:cancel', data='', headers=headers
      )

    else:
      response = requests.post(
          url=f'{job_uri.split("?")[0]}/cancel', data='', headers=headers
      )
    logging.info('Cancel response: %s', response)

  if resource_type == 'DataflowJob':
    uri_pattern = re.compile(_DATAFLOW_URI_TEMPLATE)
    match = uri_pattern.match(job_uri)
    try:
      project = match.group('project')
      location = match.group('location')
      job_id = match.group('jobid')
    except AttributeError as err:
      raise ValueError('Invalid dataflow resource URI: {}. Expect: {}.'.format(
          job_uri,
          'https://dataflow.googleapis.com/v1b3/projects/[project_id]/locations/[location]/jobs/[job_id]'
      ))

    logging.info(
        'dataflow_cancelling_job_params: %s, %s, %s', project, job_id, location
    )
    df_client = discovery.build('dataflow', 'v1b3', cache_discovery=False)
    job = (
        df_client.projects()
        .locations()
        .jobs()
        .get(projectId=project, jobId=job_id, location=location, view=None)
        .execute()
    )
    # Dataflow cancel API:
    # https://cloud.google.com/dataflow/docs/guides/stopping-a-pipeline#stopping_a_job
    logging.info('Sending Dataflow cancel request')
    job['requestedState'] = _JOB_CANCELLED_STATE

    if job['currentState'] != 'JOB_STATE_RUNNING':
      raise RuntimeError(
          'Job is not currently in running state, Job state : {}, Job needs to'
          ' be running to be cancelled'.format(job['currentState'])
      )

    logging.info('dataflow_cancelling_job: %s', job)
    job = (
        df_client.projects()
        .locations()
        .jobs()
        .update(
            projectId=project,
            jobId=job_id,
            location=location,
            body=job,
        )
        .execute()
    )
    logging.info('dataflow_cancelled_job: %s', job)
    job = (
        df_client.projects()
        .locations()
        .jobs()
        .get(projectId=project, jobId=job_id, location=location, view=None)
        .execute()
    )
    logging.info('dataflow_cancelled_job: %s', job)

  else:
    cancel_job_using_uri(job_uri, resource_type)
