from __future__ import print_function
import radical.utils as ru
from database import Database
from resource_manager import ResourceManager
from task_manager import TaskManager
from multiprocessing import Process, Event
from utils import *
from time import sleep
from exceptions import *
import traceback

class Client(object):

    """
    The Client object is instantiated by the AdaptiveMD master in order to invoke components
    on the runtime system (RTS) side. These components then interact with the MongoDB to extract
    task, resource and configuration descriptions. Using these descriptions, RTS specific
    components are created and executed.

    :arguments:
        :dburl: MongoDB URL to be used for RADICAL Pilot
        :project: Store name to be used on MongoDB. This is the store where all the configurations, resources and task
                    descriptions are present.
    """

    def __init__(self, dburl, project):

        self._uid = ru.generate_id('client.rp')
        self._logger = ru.get_logger('client.rp')

        self._dburl = dburl
        self._project = project


        # Process related data
        self._proc = None
        self._terminate = None
        self._rmgr = None
        self._tmgr = None

    # ------------------------------------------------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------------------------------------------------

    def _get_resource_desc_for_pilot(self, processed_configs, processed_resource_reqs):

        selected_resources = list()

        for resource_reqs in processed_resource_reqs:

            resource_name = resource_reqs['resource']
            #print('Resource', resource_name)

            matching_configs = get_matching_configurations(configurations=processed_configs, resource_name=resource_name)

            for matched_configs in matching_configs:
                selected_resource = dict()
                selected_resource['resource']       = str(matched_configs['resource'])
                selected_resource['runtime']        = resource_reqs['total_time']
                selected_resource['cores']          = resource_reqs['total_cpus']
                selected_resource['gpus']           = resource_reqs['total_gpus']
                selected_resource['project']        = str(matched_configs['project'])
                selected_resource['queue']          = str(matched_configs['queue'])
                selected_resource['shared_path']    = str(matched_configs.get('shared_path', '$HOME'))

            selected_resources.append(selected_resource)

        # The length of matching_configs is the number of pilots we will launch. Currently, simply return the first 
        # one.

        return selected_resources[0]

    def _runme(self):

        """
        We run the RP methods in a separate process because although the master process starts these methods,
        we don't want to block the master process.

        NOTE: DO NOT MOVE RP components outside this process. We need to keep them all inside one process since
            they processes will create new copies. Not to be run in separate threads since RP components are not 
            thread-safe.

        """

        try:

            self._db = Database(self._dburl, self._project)
            raw_resource_reqs = self._db.get_resource_requirements()
            processed_resource_reqs = process_resource_requirements(raw_resource_reqs)

            raw_configurations = self._db.get_configurations()
            processed_configurations = process_configurations(raw_configurations)

            #pprint(raw_configurations)
            #print 'Resource reqs'
            #pprint(processed_resource_reqs)
            #print 'Configs'
            #pprint(processed_configurations)

            resource_desc_for_pilot = self._get_resource_desc_for_pilot(processed_configurations, processed_resource_reqs)

            if len(resource_desc_for_pilot) > 0:
                #pprint(resource_desc_for_pilot)

                self._rmgr = ResourceManager(resource_desc = resource_desc_for_pilot, db=self._db)
                self._rmgr.submit_resource_request()

                self._tmgr = TaskManager(session=self._rmgr.session, db_obj=self._db)

                #print self._tmgr

                while not self._terminate.is_set():

                    task_descs = self._db.get_task_descriptions()

                    # print task_descs, 'while loop'

                    if task_descs:
                        cuds = create_cud_from_task_def(task_descs, self._db, resource_desc_for_pilot['shared_path'], self._project)
                        self._tmgr.run_cuds(cuds)

                        #for cud in cuds:

                            #print cud.name
                            #print cud.executable
                            #print cud.arguments
                            #print cud.input_staging
                            #print cud.output_staging
                            #print cud.cores

                            #sleep(3)

                    else:
                        sleep(3)

            else:
                raise Error(msg="No matching resource found in configuration file. Please check your configuration file and the resource object.")

        except Exception as ex:

            self._logger.error('Client process failed, error: %s'%ex)
            print(traceback.format_exc())
            raise Error(msg=ex)

        finally:

            if self._rmgr:
                self._rmgr.pilot.cancel()
                self._rmgr.session.close(cleanup=False)

    # ------------------------------------------------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------------------------------------------------

    def start(self):

        """
        Method exposed to the master process to start all the RP components and processes. This is a non-blocking command.
        """

        try:
            self._proc = Process(target=self._runme, args=())
            self._terminate = Event()
            self._proc.start()

        except Exception as ex:

            self._logger.error("Error starting RP process, error: %s"%ex)
            self.end()
            raise Error(msg=ex)

    def stop(self):

        """
        Method exposed to the master process to end all the RP components and processes. This is a blocking command.
        """

        try:

            if self._proc.is_alive():

                if not self._terminate.is_set():
                    self._terminate.set()

                self._proc.join()

        except Exception as ex:

            self._logger.error("Error stopping RP process, error: %s"%ex)
            raise Error(msg=ex)

    # ------------------------------------------------------------------------------------------------------------------
