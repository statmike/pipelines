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
"""Test Vertex AI Custom Job Remote Runner Client module."""

import json
from logging import raiseExceptions
import os
import time
import requests
import google.auth
import unittest
from google.cloud import aiplatform
from unittest import mock
from google.cloud.aiplatform.compat.types import job_state as gca_job_state
from google_cloud_pipeline_components.container.utils.execution_context import ExecutionContext
from google_cloud_pipeline_components.container.experimental.cancel_job import remote_runner as cancel_job_remote_runner
from google_cloud_pipeline_components.container.v1.wait_gcp_resources import remote_runner as wait_gcp_resources_remote_runner
import googleapiclient.discovery as discovery
from google_cloud_pipeline_components.container.v1.custom_job import remote_runner as custom_job_remote_runner

_SUPPORTED_JOB_TYPES = ['DataflowJob',
                        'VertexLro',
                        'BigQueryJob',
                        'BatchPredictionJob',
                        'HyperparameterTuningJob',
                        'CustomJob',
                        'DataprocLro']


class CancelJobTests(unittest.TestCase):

  def setUp(self):
    super(CancelJobTests, self).setUp()
    self._dataflow_payload = (
        '{"resources": [{"resourceType": "DataflowJob",'
        '"resourceUri": "https://dataflow.googleapis.com/'
        'v1b3/projects/foo/locations/us-central1/jobs/job123"}]}'
    )

    self._custom_job_payload = (
        '{"resources": [{"resourceType": "CustomJob",'
        '"resourceUri": "https://us-aiplatform.googleapis.com/v1/test_job"}]}'
    )

    self._big_query_payload = (
        '{"resources": [{"resourceType": "CustomJob",'
        '"resourceUri": "https://www.googleapis.com/bigquery/v2/projects/'
        'test_project/jobs/fake_job?location=US"}]}'
    )

    self._project = 'project1'
    self._location = 'us-central1'
    self._gcp_resources_path = os.path.join(
        os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), 'gcp_resources')
    self._type = 'DataflowJob'
    self._custom_job_name = 'test_job'
    self._custom_job_uri_prefix = 'https://us-aiplatform.googleapis.com/v1/'

  def tearDown(self):
    if os.path.exists(self._gcp_resources_path):
      os.remove(self._gcp_resources_path)

  @mock.patch.object(discovery, 'build', autospec=True)
  @mock.patch.object(ExecutionContext, '__init__', autospec=True)
  def test_cancel_job_on_non_completed_data_flow_job_suceeds(
      self, mock_execution_context, mock_build
  ):
    df_client = mock.Mock()
    mock_build.return_value = df_client
    expected_done_job = {'id': 'job-1', 'currentState': 'JOB_STATE_DONE'}
    expected_running_job = {'id': 'job-1', 'currentState': 'JOB_STATE_RUNNING'}
    get_request = mock.Mock()
    df_client.projects().locations().jobs().get.return_value = get_request
    # The first done_job is to end the polling loop.
    get_request.execute.side_effect = [
        expected_done_job,
        expected_running_job,
        expected_running_job,
    ]
    update_request = mock.Mock()
    df_client.projects().locations().jobs().update.return_value = update_request
    mock_execution_context.return_value = None
    expected_cancel_job = {
        'id': 'job-1',
        'currentState': 'JOB_STATE_RUNNING',
        'requestedState': 'JOB_STATE_CANCELLED',
    }

    wait_gcp_resources_remote_runner.wait_gcp_resources(
        self._type,
        self._project,
        self._location,
        self._dataflow_payload,
        self._gcp_resources_path,
    )

    cancel_job_remote_runner.cancel_job(
        self._dataflow_payload,
    )

    df_client.projects().locations().jobs().update.assert_called_once_with(
        projectId='foo',
        jobId='job123',
        location='us-central1',
        body=expected_cancel_job,
    )

  @mock.patch.object(discovery, 'build', autospec=True)
  def test_cancel_job_on_completed_dataflow_fails(self, mock_build):
    # test on succeeded job
    df_client = mock.Mock()
    mock_build.return_value = df_client
    expected_job = {'id': 'job-1', 'currentState': 'JOB_STATE_DONE'}
    get_request = mock.Mock()
    df_client.projects().locations().jobs().get.return_value = get_request
    get_request.execute.return_value = expected_job

    with self.assertRaises(RuntimeError):
      cancel_job_remote_runner.cancel_job(
          self._dataflow_payload,
      )

    # test on failed job
    df_client = mock.Mock()
    mock_build.return_value = df_client
    expected_job = {'id': 'job-1', 'currentState': 'JOB_STATE_FAILED'}
    get_request = mock.Mock()
    df_client.projects().locations().jobs().get.return_value = get_request
    get_request.execute.return_value = expected_job

    with self.assertRaises(RuntimeError):
      cancel_job_remote_runner.cancel_job(
          self._dataflow_payload,
      )

  @mock.patch.object(google.auth, 'default', autospec=True)
  @mock.patch.object(discovery, 'build', autospec=True)
  @mock.patch.object(requests, 'post', autospec=True)
  @mock.patch.object(requests, 'get', autospec=True)
  def test_cancel_job_on_custom_job_suceeds(
      self,
      mock_get_requests,
      mock_post_requests,
      mock_build,
      mock_auth,
  ):
    creds = mock.Mock()
    creds.token = 'fake_token'
    mock_auth.return_value = [creds, 'project']
    mock_auth.return_value = [creds, 'project']
    client = mock.Mock()
    mock_build.return_value = client
    expected_job = {'id': 'job-1', 'currentState': 'JOB_STATE_RUNNING'}
    get_request = mock.Mock()
    client.projects().locations().jobs().get.return_value = get_request
    get_request.execute.return_value = expected_job

    cancel_job_remote_runner.cancel_job(
        self._custom_job_payload,
    )

  @mock.patch.object(aiplatform.gapic, 'JobServiceClient', autospec=True)
  @mock.patch.object(google.auth, 'default', autospec=True)
  @mock.patch.object(google.auth.transport.requests, 'Request', autospec=True)
  @mock.patch.object(requests, 'post', autospec=True)
  @mock.patch.object(ExecutionContext, '__init__', autospec=True)
  def test_custom_job_remote_runner_cancel(
      self,
      mock_execution_context,
      mock_post_requests,
      _,
      mock_auth,
      mock_job_service_client,
  ):
    creds = mock.Mock()
    creds.token = 'fake_token'
    mock_auth.return_value = [creds, 'project']

    job_client = mock.Mock()
    mock_job_service_client.return_value = job_client

    create_custom_job_response = mock.Mock()
    job_client.create_custom_job.return_value = create_custom_job_response
    create_custom_job_response.name = self._custom_job_name

    get_custom_job_response = mock.Mock()
    job_client.get_custom_job.return_value = get_custom_job_response
    get_custom_job_response.state = gca_job_state.JobState.JOB_STATE_SUCCEEDED
    mock_execution_context.return_value = None

    custom_job_remote_runner.create_custom_job(
        self._type,
        self._project,
        self._location,
        self._custom_job_payload,
        self._gcp_resources_path,
    )

    # Call cancellation handler
    cancel_job_remote_runner.cancel_job(
        self._custom_job_payload,
    )

    mock_post_requests.assert_called_once_with(
        url=f'{self._custom_job_uri_prefix}{self._custom_job_name}:cancel',
        data='',
        headers={
            'Content-type': 'application/json',
            'Authorization': 'Bearer fake_token',
        },
    )

    # add some asserts to show that request was called with the right parameters
    # and request response is good

    # for already completed custom job
    # with self.assertRaises(RuntimeError):
    #   cancel_job_remote_runner.cancel_job(
    #       self._custom_job_payload,
    #   )

  @mock.patch.object(discovery, 'build', autospec=True)
  def test_cancel_job_on_failed_dataflow_fails(self, mock_build):
    df_client = mock.Mock()
    mock_build.return_value = df_client
    expected_job = {'id': 'job-1', 'currentState': 'JOB_STATE_FAILED'}
    get_request = mock.Mock()
    df_client.projects().locations().jobs().get.return_value = get_request
    get_request.execute.return_value = expected_job

    with self.assertRaises(RuntimeError):
      cancel_job_remote_runner.cancel_job(
          self._dataflow_payload,
      )

  @mock.patch.object(discovery, 'build', autospec=True)
  def test_cancel_job_on_invalid_gcp_resource_type_fails(self, mock_build):
    invalid_payload = (
        '{"resources": [{"resourceType": "InvalidType",'
        '"resourceUri": "https://dataflow.googleapis.com/v1b3/'
        'projects/foo/locations/us-central1/jobs/job123"}]}'
    )
    with self.assertRaises(ValueError):
      cancel_job_remote_runner.cancel_job(
          invalid_payload,
      )

  @mock.patch.object(discovery, 'build', autospec=True)
  def test_cancel_job_on_empty_gcp_resource_fails(self, mock_build):
    invalid_payload = '{"resources": [{}]}'
    with self.assertRaises(ValueError):
      cancel_job_remote_runner.cancel_job(
          invalid_payload,
      )

  @mock.patch.object(discovery, 'build', autospec=True)
  def test_cancel_job_on_invalid_gcp_resource_uri_fails(self, mock_build):
    invalid_payload = (
        '{"resources": [{"resourceType": "DataflowJob",'
        '"resourceUri": "https://dataflow.googleapis.com/'
        'v1b3/projects/abc"}]}'
    )
    with self.assertRaises(ValueError):
      cancel_job_remote_runner.cancel_job(
          invalid_payload
      )
