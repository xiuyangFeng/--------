import numpy as np
import pytest
from wss_v5.anatomy_overrides import correct_topology, correct_openings
from wss_v5.geometry_candidate import topology_semantics, plain

def source():
    parents=[-1,0,0,2,2,1,1]
    names=['','','','out-li','out-re','out-ri','out-le']
    segments=[{'segment_id':i,'parent_id':p,'outlet_name':names[i]} for i,p in enumerate(parents)]
    points=np.array([[i,0,z] for i in range(7) for z in (0,1)],float)
    return topology_semantics(segments,points,np.repeat(np.arange(7),2),np.tile([0,1],7))

def test_confirmed_names_preserve_original_correspondence_and_scope():
    s,t=source();a,b=correct_topology('AAA/ruputer/WANG_KUI_WU',s,t)
    assert [x['anatomy_label'] for x in a]==['trunk','left_cia','right_cia','right_internal','right_external','left_internal','left_external']
    assert [x['cfd_zone_label'] for x in a]==[x['cfd_zone_label'] for x in s]
    assert [x['segment_id'] for x in a if x['anatomy_label_user_confirmed']]==[3,5]
    assert b['mixed_side_cia_segments']==[] and b['original_CFD_mixed_side_cia_segments']==[1,2]
    assert plain(correct_topology('AAA/ruputer/WANG_KUI_WU',a,b))==plain((a,b))
    o=correct_openings('AAA/ruputer/WANG_KUI_WU',[{'segment_id':3,'label_provisional':'out-li'}],a)[0]
    assert o['label_provisional']=='out-li' and o['anatomy_outlet_label']=='out-ri'

def test_override_does_not_apply_to_other_case_or_changed_tree():
    s,t=source()
    assert correct_topology('other',s,t)==(s,t)
    s[3]['parent_id']=1
    with pytest.raises(ValueError,match='expected segment tree'):
        correct_topology('AAA/ruputer/WANG_KUI_WU',s,t)
