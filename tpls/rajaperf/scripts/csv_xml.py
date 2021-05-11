#!/bin/env python

import csv
from datetime import datetime
import os
import xml.etree.ElementTree as ET
import xml

# https://stackabuse.com/reading-and-writing-xml-files-in-python/

# xmlformatter:
# https://www.freeformatter.com/xml-formatter.html#ad-output


infile = "./RAJAPerf-timing.csv"

def read_infile(infile):
    """STUB"""
    with open(infile) as csvfile:
        rps_reader = csv.reader(csvfile, delimiter=',')



def get_date():
    """STUB"""
    date = datetime.now().strftime("%-Y-%m-%dT%H:%M:%S")
    return date


date = get_date()

perf_report = ET.Element("performance-report")

name ="RAJAPerf" + date + ".xml"

time_units="seconds"

perf_report.set("date", date)

perf_report.set("name", name)

perf_report.set("time-units", time_units)

perf_root = ET.SubElement(perf_report, 'timing')

perf_root.set("end-time",date)

perf_root.set("name", "kokkos_perf_suite")

#print(ET.tostring(perf_report))

# b'<performance-report time-units="seconds" date="2020-12-16T14:34:40"
# name="RAJAPerf-timing.csv"><timing end-time="2020-12-16T14:34:40"
# name="kokkos_perf_suite" /></performance-report>'

# metadata TBD

# create hierarchy

test_suite_list = []
with open(infile) as csvfile:
    rps_reader = csv.reader(csvfile, delimiter=',')
    for row in rps_reader:
        test_suite_list.append(row)


suite_names_set = set([x[0][:x[0].find("_")] for x in test_suite_list[2:]])

#suite_names_set
#Out[135]: {'Basic', 'KokkosMechanics'}


heirarch_dict = dict()
for name in suite_names_set:
    heirarch_dict[name] = []

# heirarch_dict
# Out[137]: {'KokkosMechanics': [], 'Basic': []}

for item in test_suite_list[2:]:
    key = item[0][:item[0].find("_")]
    heirarch_dict[key].append(item)
    #print(item)

#NEXT STEPS:  For the main test categories, Basic and KokkosMechanics, sum
# the test times over all of the kernels for each of their variants

col_meanings_dict = dict()

for index, item in enumerate(test_suite_list[1]):
    #print(index, item)
    col_meanings_dict[index] = item

#col_meanings_dict
# Out[152]:
# {0: 'Kernel                         ',
#  1: ' Base_Seq ',
#  2: ' Lambda_Seq ',
#  3: ' RAJA_Seq ',
#  4: ' Base_CUDA ',
#  5: ' RAJA_CUDA ',
#  6: ' Kokkos_Lambda_Seq ',
#  7: ' Kokkos_Functor_Seq ',
#  8: ' Kokkos_Lambda_CUDA ',
#  9: ' Kokkos_Functor_CUDA'}


def associate_timings_with_xml(xml_element, timing_dict, suite_or_test_name):
    """STUB -- xml_element will be an element of perf_report;
    timing_dict = a map of variant names to test run times
    """
    for key, value in timing_dict.items():
        xml_element.set(key.lower(), str(value))
    xml_element.set("name", suite_or_test_name.strip())



def create_RPS_xml_report(suite_name, suite_data_list):
    """STUB - suite_name is a string = Basic, KokkosMechanics, etc.;
    suite_data_list will be the values for a key, Basic or KokkosMechanics
    """
    aggregate_results_dict = dict()
    #print(suite_data_list)
    for list_item in suite_data_list:
        for index, timing in enumerate(list_item[1:]):
            if "Not run" in timing:
                continue
            variant_name = col_meanings_dict[index + 1]
            if variant_name not in aggregate_results_dict:
                aggregate_results_dict[variant_name] = 0.0
            # sums values of all the basic kernels
            aggregate_results_dict[variant_name] += float(timing)
    #print(aggregate_results_dict)

    suite_root = ET.SubElement(perf_root, "timing")
    associate_timings_with_xml(suite_root, aggregate_results_dict, suite_name)
    for list_item in suite_data_list:
        test_timings_dict = dict()
        for index, timing in enumerate(list_item[1:]):
            if "Not run" in timing:
                continue
            variant_name = col_meanings_dict[index + 1]
            test_timings_dict[variant_name] = float(timing)
        xml_element_for_a_kernel_test = ET.SubElement(suite_root, "timing")
        associate_timings_with_xml(xml_element_for_a_kernel_test,
test_timings_dict, list_item[0])



def run():
    """STUB"""

    read_infile(infile)

    #create_RPS_xml_report("Basic", heirarch_dict["Basic"])

    for key in heirarch_dict.keys():
        create_RPS_xml_report(key, heirarch_dict[key])

	# Aided in debugging
    #print(heirarch_dict["KokkosMechanics"])

    # Prints xml to screen as string
	#print(ET.tostring(perf_report))

    ET.dump(perf_report)




if __name__ == "__main__":
    run()

