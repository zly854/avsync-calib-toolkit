"""Regression checks for published controls and failure-prone input boundaries."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from av_align_curve import calc_intersection_over_union, iou_in_window
from make_table2 import MAGS, block_boot_separable

class ReleaseTests(unittest.TestCase):
    def test_one_to_one_and_strict_boundary(self):
        self.assertEqual(calc_intersection_over_union([.99,1.01],[1.0],25),.5)
        self.assertEqual(calc_intersection_over_union([.96,1.04],[1.0],25),0)
        self.assertEqual(calc_intersection_over_union([1.0],[],25),0)
        self.assertTrue(np.isnan(calc_intersection_over_union([],[],25)))
        with self.assertRaises(ValueError):calc_intersection_over_union([1],[1],0)
    def test_cache_control(self):
        caches=[json.loads(p.read_text()) for p in sorted((ROOT/'data/peaks_cache').glob('*_peaks.json'))]
        self.assertEqual(len(caches),14)
        matched=[iou_in_window(p['apk'],p['vpk'],p['fps'],0,p['dur'])[0] for p in caches]
        self.assertAlmostEqual(float(np.mean(matched)),.1902,places=4)
        mismatch=[iou_in_window(a['apk'],v['vpk'],v['fps'],0,min(a['dur'],v['dur']))[0] for i,a in enumerate(caches) for j,v in enumerate(caches) if i!=j]
        self.assertEqual(len(mismatch),182)
        self.assertAlmostEqual(float(np.mean(mismatch)),.1907,places=4)
    def test_block_bootstrap_published_counts(self):
        adv=np.zeros(8,dtype=int);delay=np.zeros(8,dtype=int)
        paths=sorted((ROOT/'data').glob('ref*_desync_calib_ext.json'));self.assertEqual(len(paths),14)
        for p in paths:
            curves=json.loads(p.read_text())['curves'];zero=np.array(curves['+0.00']['expect'])
            for i,m in enumerate(MAGS):
                for sign,result in [('+',adv),('-',delay)]:
                    result[i]+=block_boot_separable(np.array(curves[sign+m]['expect']),zero)
        self.assertEqual(adv.tolist(),[3,3,6,9,12,12,8,5])
        self.assertEqual(delay.tolist(),[5,5,5,7,12,12,6,3])
    def test_manifest_and_dry_run(self):
        meta=json.loads((ROOT/'data/refset_meta.json').read_text())
        self.assertEqual([r['name'] for r in meta],[f'ref{i:02}' for i in range(14)])
        self.assertEqual([sum(r['category']==k for r in meta) for k in ['speech','music','impact','ambient']],[4,4,4,2])
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'not-created'
            p=subprocess.run([sys.executable,str(ROOT/'scripts/build_refset.py'),'--out',str(out),'--dry-run'],capture_output=True,text=True,check=True)
            self.assertEqual(p.stdout.count('curl --fail'),14)
            self.assertFalse(out.exists())
    def test_missing_reference_data_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=subprocess.run([sys.executable,str(ROOT/'scripts/make_table2.py'),'--data-dir',tmp],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('Expected 14',p.stderr)

if __name__=='__main__': unittest.main()
