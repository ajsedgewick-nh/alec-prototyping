from xml.dom import minidom
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn import metrics


def dist2aff(dmat, thresh=100):
    #thresh = 100
    print(thresh)
    outmat = dist_thresh(dmat, thresh)
    outmat = 1 - (outmat/thresh)
    
    return outmat

def dist_thresh(dmat, thresh=100):
    #thresh = 100
    print(thresh)
    outmat = dmat.copy()
    outmat[outmat > thresh] = thresh

    return outmat

# write cluster assignments from pandas to xml that can be scored by ALEC
def cluster2xml(clustdf, fn):
    clusts = np.unique(clustdf['cluster'])
    
    sits = ET.Element('situations')
    sits.set('xmlns', 'http://xmlns.opennms.org/xsd/alec/model/v1.0.0')
    
    for cl in clusts:
        cursit = ET.SubElement(sits, 'situation')
        cursit.set('id', 'cluster_' + str(cl))
        
        alarms = clustdf.loc[clustdf['cluster']==cl, 'id']
        for al in alarms:
            cural = ET.SubElement(cursit, 'alarm-ref')
            cural.set('id', str(al))
            
    #root = ET.ElementTree(sits)
    xmlstr = minidom.parseString(ET.tostring(sits)).toprettyxml(indent="    ")
    with open(fn, "w") as f:
        f.write(xmlstr)
    #return ET.tostring(sits, method='xml')
    return #xmlstr

# read situations XML output from ALEC into a pandas dataframe
def read_xml_clusters(fn):
    tree = ET.parse(fn)
    root = tree.getroot()
    clust_list = []
    id_list = []
    for situation in root:
        cur_clust = situation.attrib['id']
        for alarm in situation:
            clust_list.append(cur_clust)
            id_list.append(alarm.attrib['id'])
            
    return pd.DataFrame({'id':id_list, 'cluster':clust_list})

# helper for mapping each alarm to a list of its peers for a given clustering
def map_peers(clustdf):
    clust_dict = dict()
    alarm_dict = dict()
    
    for index, row in clustdf.iterrows():
        cur_clust = str(row['cluster'])
        cur_id = str(row['id'])
        alarm_dict[cur_id] = cur_clust
        if cur_clust not in clust_dict:
            clust_dict[cur_clust] = set()
        clust_dict[cur_clust].add(cur_id)
        
    for k,v in alarm_dict.items():
        alarm_dict[k] = clust_dict[v]
        
    return alarm_dict

# implementation of the "peer" scoring strategy built into ALEC
def peer_score(clust1, clust2):
    map1 = map_peers(clust1)
    map2 = map_peers(clust2)
    
    score_dict = {'score':0.0, 'exact':0, 'partial':0, 'mismatch':0}

    for k,v in map1.items():
        v2 = map2.get(k, [])
        # alarm not found in second clustering 
        if len(v2) == 0:
            continue
            
        if len(v) == 1 and len(v2) == 1:
            cur_score = 1.0
        else:
            cur_score = (len(v.intersection(v2)) - 1) / (max(len(v), len(v2)) - 1)
            
            
        if len(v.intersection(v2)) == 0:
            print(k, v, v2)
            
        score_dict['score'] += cur_score
        if cur_score == 0:
            score_dict['mismatch'] += 1
        elif cur_score < 1:
            score_dict['partial'] += 1
        elif cur_score == 1.0:
            score_dict['exact'] += 1
    
    score_dict['pct'] = score_dict['score'] / len(map1) * 100
    return score_dict

# given a gold standard clustering (refclust), score a candidate clustering (predclust)
# using peer, ARI and AMI
def score_clusters(refclust, predclust):#, scoreid=None):
    join_clust = refclust.set_index('id').join(predclust.set_index('id'), lsuffix='ref', rsuffix='pred')
    join_clust = join_clust.loc[pd.notna(join_clust['clusterpred']),:]
    print(join_clust.shape)
    
    
    resdict = {}
    #if scoreid is not None:
    #    resdict['scoreid'] = scoreid
    
    resdict['adjRand'] = metrics.adjusted_rand_score(join_clust['clusterref'], 
                                                     join_clust['clusterpred'])
    resdict['adjMI'] = metrics.adjusted_mutual_info_score(join_clust['clusterref'], 
                                                          join_clust['clusterpred'])
    resdict['peer'] = peer_score(refclust, predclust)['pct'] / 100.0
    return resdict

# parse useful info from alarm XML file into pandas
def read_alarm_xml(fn):
    with open(fn, 'r') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        alarmlist = []
        for alarm in root:
            alarmlist.append({'alarm-id':alarm.attrib['id'],
                              'object-id':alarm.attrib['inventory-object-id'],
                              'object-type':alarm.attrib['inventory-object-type'],
                              'first-event-time':int(alarm.attrib['first-event-time']),
                              'last-event-time':int(alarm.attrib['last-event-time']),
                              'duration':(int(alarm.attrib['last-event-time']) - int(alarm.attrib['first-event-time'])),
                              'summary':alarm.attrib['summary']})
    
        return pd.DataFrame(alarmlist)

    
# parse useful info from alarm XML file into pandas
def read_alarm_xml_events(fn):
    with open(fn, 'r') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        eventlist = []
        for alarm in root:
            for event in alarm:
                eventlist.append({'event-id':event.attrib['id'],
                                  'alarm-id':alarm.attrib['id'],
                                  'object-id':alarm.attrib['inventory-object-id'],
                                  'object-type':alarm.attrib['inventory-object-type'],
                                  'event-time':int(event.attrib['time']),
                                  'event-severity':event.attrib['severity'],
                                  'event-summary':event.attrib['summary'],
                                  'alarm-summary':alarm.attrib['summary']})
    
        return pd.DataFrame(eventlist)