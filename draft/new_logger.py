from bacpypes.service.device import LocalDeviceObject
from bacpypes.basetypes import ServicesSupported
from bacpypes.app import BIPSimpleApplication
from bacpypes.iocb import IOCB
from bacpypes.apdu import WhoIsRequest, IAmRequest, ReadPropertyRequest
from bacpypes.errors import DecodingError
import sys
from bacpypes.debugging import  ModuleLogger
from bacpypes.consolelogging import ConfigArgumentParser
from bacpypes.core import run, stop
from bacpypes.pdu import Address
from bacpypes.object import get_datatype
from datetime import datetime, timedelta
from collections import OrderedDict

import os
import django
import pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")
sys.path.insert(0, "/var/www/gismap/")
sys.path.append('/your/main/project/directory')
#print (sys.path)
django.setup()
from bacnet.models import *

_debug=1
_log = ModuleLogger(globals())

object_identifier = None
object_destination = None
this_application = None
value = None
class GetNAEInfoApplication(BIPSimpleApplication):
    def __init__(self, device, address):
        BIPSimpleApplication.__init__(self, device, address)
        self._request=None
        iocb = IOCB(time=5)

    def request(self, apdu):
        print(apdu)
        self._request=apdu
        BIPSimpleApplication.request(self, apdu)

    def confirmation(self, apdu):
        global value
        datatype = get_datatype(apdu.objectIdentifier[0], apdu.propertyIdentifier)
        value=apdu.propertyValue.cast_out(datatype)
        BIPSimpleApplication.confirmation(self, apdu)
        stop()

    def indication(self, apdu):
        global object_identifier
        global object_destination
        if (isinstance(self._request, WhoIsRequest)) and (isinstance(apdu, IAmRequest)):
            device_type, device_instance = apdu.iAmDeviceIdentifier
            if device_type != 'device':
                raise DecodingError("invalid object type")
            if (self._request.deviceInstanceRangeLowLimit is not None) and \
                (device_instance < self._request.deviceInstanceRangeLowLimit):
                pass
            elif (self._request.deviceInstanceRangeHighLimit is not None) and \
                (device_instance > self._request.deviceInstanceRangeHighLimit):
                pass
            else:
                object_identifier=apdu.iAmDeviceIdentifier
                object_destination = apdu.pduSource
                sys.stdout.write('pduSource = ' + repr(apdu.pduSource) + '\n')
                sys.stdout.write('iAmDeviceIdentifier = ' + str(apdu.iAmDeviceIdentifier) + '\n')
                sys.stdout.write('maxAPDULengthAccepted = ' + str(apdu.maxAPDULengthAccepted) + '\n')
                sys.stdout.write('segmentationSupported = ' + str(apdu.segmentationSupported) + '\n')
                sys.stdout.write('vendorID = ' + str(apdu.vendorID) + '\n')
                if _debug: GetNAEInfoApplication._debug("__init__ %r", apdu)
                sys.stdout.flush()

        BIPSimpleApplication.indication(self, apdu)
        stop()
def findNAE():
    global object_identifier
    global object_destination
    global this_application
    args = ConfigArgumentParser(description=__doc__).parse_args()
    this_device = LocalDeviceObject(objectName=args.ini.objectname,
                                    objectIdentifier=int(args.ini.objectidentifier),
                                    maxApduLengthAccepted=int(args.ini.maxapdulengthaccepted),
                                    segmentationSupported=args.ini.segmentationsupported,
                                    vendorIdentifier=int(args.ini.vendoridentifier))

    this_application = GetNAEInfoApplication(this_device, args.ini.address)
    pss = ServicesSupported()
    pss['whoIs'] = 1
    pss['iAm'] = 1
    pss['readProperty'] = 1
    pss['writeProperty'] = 1
    this_device.protocolServicesSupported = pss
    services_supported = this_application.get_services_supported()
    this_device.protocolServicesSupported = services_supported.value
    this_application.who_is(None, None, Address(b'\x0A\x0C\x00\xFA\xba\xc0'))

    run()


count = 0


def write(name, value, school):
    global count
    global csv
    csv = open('trend.csv', 'a')
    csv.write(str(count) + "," + str(datetime.now()) + "," + name + "," + str(value) + "," + school + ",\n")
    csv.close()
    count += 1

def analog_value_request(identifier, name, school):
    global object_destination
    global value
    new_request = ReadPropertyRequest(objectIdentifier=("analogInput", identifier), propertyIdentifier="presentValue")
    new_request.pduDestination = object_destination
    this_application.request(new_request)
    run()
    dv = Data_Point(Value=value, Time = str(datetime.now()), Name = name, School = School.objects.get(Name=school))
    dv.save()
    write(name, value, school)

def close():
    global csv
    csv.close()

def get_value(identifier):
    """Request the current analog value of the name/ID property pair

    Parameters
    ----------
    identifier : int
        The identifier number of the sensor from which the
        reading will be taken
    """
    global object_destination
    global value
    new_request = ReadPropertyRequest(
        objectIdentifier=("analogInput", identifier),
        propertyIdentifier="presentValue")
    new_request.pduDestination = object_destination
    this_application.request(new_request)
    run()
    return value

def round_datetime(dt, res):
    """Rounds a datetime object (up or down) to the desired timedelta resolution

    Parameters
    ----------
    dt : datetime
        Datetime to be rounded
    res : timedelta
        Desired rounding resolution

    Returns
    -------
    dt_rounded : datetime
        Rounded datetime
    """
    td = dt - datetime.min
    td_round = res*round(td/res)
    return td_round + datetime.min

def init_csv(props):
    """ Initialize the trend CSV file with header labels

    Parameters
    ----------
    props : DataFrame
        Property table with a 'Name' column to use as log headers
    """
    with open('trend2.csv','w') as csv:
        csv.write(','.join(['Timestamp']+list(props['Name'])) + '\n')

def write_prop_values(props):
    """Write all the property values to the trend file
    at the current (rounded) time

    Parameters
    ----------
    props : DataFrame
        Property table, must contain an 'Identifier' column
    """
    resp = [str(get_value(id_)) for id_ in props['Identifier']]
    now = round_datetime(datetime.now(),timedelta(seconds=1))
    with open('trend2.csv', 'a') as csv:
        csv.write(','.join([str(now)]+resp) + '\n')


