#!/usr/bin/env python3
"""Independent MCAP -> sampled RGB JPEG input preparation. No model or GPU usage."""
import argparse, bisect, hashlib, io, json, platform, statistics, struct, sys, time
from collections import Counter
from fractions import Fraction
from pathlib import Path
import av
import mcap
from mcap.reader import make_reader
from PIL import Image, __version__ as pillow_version

QUALITY = 80

def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

class CDR:
    def __init__(self, data):
        if data[:4] != b'\x00\x01\x00\x00':
            raise ValueError('Only little-endian CDR1 is supported')
        self.b=data; self.p=4
    def u32(self, signed=False):
        self.p += (-(self.p-4)) % 4
        x=struct.unpack_from('<i' if signed else '<I',self.b,self.p)[0]; self.p+=4
        return x
    def string(self):
        n=self.u32(); x=self.b[self.p:self.p+n]; self.p+=n
        if len(x)!=n or not x.endswith(b'\0'): raise ValueError('Malformed CDR string')
        return x[:-1].decode()
    def header(self):
        sec=self.u32(True); ns=self.u32()
        if not 0<=ns<10**9: raise ValueError('Invalid stamp')
        return sec*10**9+ns,self.string()
    def image(self):
        stamp,frame_id=self.header(); fmt=self.string(); n=self.u32()
        payload=self.b[self.p:self.p+n]
        tail=self.b[self.p+n:]
        if len(payload)!=n or len(tail)>3 or any(tail): raise ValueError('Truncated/non-padding trailing image payload')
        return stamp,frame_id,fmt,payload

def messages(paths, topic):
    for path in paths:
        with path.open('rb') as f:
            for schema,ch,msg in make_reader(f,validate_crcs=True).iter_messages(topics=[topic]):
                if ch.message_encoding!='cdr' or schema.name!='sensor_msgs/msg/CompressedImage':
                    raise ValueError('Expected CDR sensor_msgs/msg/CompressedImage')
                yield path,schema,ch,msg

def nal_types(payload):
    # Annex B start codes, retained as source evidence rather than assuming topic suffix.
    import re
    return [payload[m.end()] & 31 for m in re.finditer(b'\x00\x00\x01',payload) if m.end()<len(payload)]

def inspect_source(paths,topic):
    started=time.perf_counter(); parts=[]; all_topics=Counter(); metadata=[]; formats=Counter(); ids=Counter()
    for path in paths:
        with path.open('rb') as f:
            s=make_reader(f,validate_crcs=True).get_summary()
            if not s or not s.statistics: raise ValueError('MCAP summary required')
            topics={c.topic:s.statistics.channel_message_counts.get(c.id,0) for c in s.channels.values()}
            all_topics.update(topics)
            parts.append({'name':path.name,'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,
                          'topics':topics,'start_log_ns':s.statistics.message_start_time,'end_log_ns':s.statistics.message_end_time})
    first_payload=None
    for path,schema,ch,msg in messages(paths,topic):
        stamp,fid,fmt,payload=CDR(msg.data).image()
        if first_payload is None: first_payload=payload
        formats[fmt]+=1; ids[fid]+=1
        metadata.append({'index':len(metadata),'part':path.name,'stamp_ns':stamp,'log_ns':msg.log_time,
                         'publish_ns':msg.publish_time,'sequence':msg.sequence,'payload_bytes':len(payload)})
    stamps=[m['stamp_ns'] for m in metadata]
    if len(stamps)<2 or any(b<=a for a,b in zip(stamps,stamps[1:])): raise ValueError('Need strictly increasing acquisition timestamps')
    if set(formats)!={'h264'} or not {7,8,5}.issubset(set(nal_types(first_payload))):
        raise ValueError('Unsupported stream: expected verified Annex B H264 with SPS/PPS/IDR')
    period=round(statistics.median(b-a for a,b in zip(stamps,stamps[1:])))
    probe=av.CodecContext.create('h264','r'); probe.thread_count=1
    frames=probe.decode(av.Packet(first_payload))+probe.decode(None)
    if len(frames)!=1: raise ValueError('First packet does not decode to one frame')
    dims=[frames[0].width,frames[0].height]
    camera_info=[]
    for path in paths:
        with path.open('rb') as f:
            for schema,ch,msg in make_reader(f).iter_messages(topics=['/camera/camera/color/camera_info']):
                c=CDR(msg.data); stamp,fid=c.header(); h=c.u32(); w=c.u32()
                camera_info.append({'part':path.name,'frame_id':fid,'width':w,'height':h,'stamp_ns':stamp}); break
    info={'parts':parts,'topic_counts':dict(all_topics),'selected_topic':topic,'format_counts':dict(formats),
          'frame_id_counts':dict(ids),'first_payload_nal_types':nal_types(first_payload), 'codec':'h264',
          'decoded_dimensions':dims,'camera_info_samples':camera_info,'rgb_messages':len(metadata),
          'first_stamp_ns':stamps[0],'last_stamp_ns':stamps[-1],'median_period_ns':period,
          'timestamp_span_s':(stamps[-1]-stamps[0])/1e9,'coverage_duration_s':(stamps[-1]-stamps[0]+period)/1e9,
          'duration_definition':'last-first + median frame period; window origin is first RGB header acquisition stamp',
          'inspection_s':time.perf_counter()-started}
    return info,metadata

def plan_windows(stamps,period_ns,window_s,interval_s):
    if not stamps or interval_s<=0 or window_s<=0 or interval_s>window_s: raise ValueError('Invalid sampling parameters')
    base=stamps[0]; end=stamps[-1]+period_ns; window=round(window_s*1e9); step=round(interval_s*1e9)
    result=[]
    for start in range(base,end,window):
        stop=min(start+window,end); lo=bisect.bisect_left(stamps,start); hi=bisect.bisect_left(stamps,stop)
        if hi==lo: result.append({'start_ns':start,'end_exclusive_ns':stop,'indices':[]}); continue
        selected=[]; targets=[]
        for target in range(start,stop,step):
            idx=bisect.bisect_left(stamps,target,lo,hi)
            if idx<hi:
                targets.append({'target_ns':target,'index':idx,'actual_stamp_ns':stamps[idx]})
                selected.append(idx)
        selected.append(hi-1)
        result.append({'start_ns':start,'end_exclusive_ns':stop,'indices':sorted(set(selected)),
                       'targets':targets,'last_frame_index':hi-1})
    return result

def target_size(width,height,longest):
    scale=min(1.0,longest/max(width,height)) if longest else 1.0
    return max(1,round(width*scale)),max(1,round(height*scale))

def jpeg_bytes(image,quality=QUALITY):
    stream=io.BytesIO(); image.save(stream,format='JPEG',quality=quality,subsampling=2,optimize=False)
    return stream.getvalue()

def consume_jpeg_path(path):
    """Decode the pinned JPEG only when inference requests its pixels."""
    with Image.open(path) as image:
        image.load()
        return image.convert('RGB')

def run(paths,topic,out,interval=.5,longest=960,window=3,threads=1,quality=QUALITY):
    total_start=time.perf_counter(); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    save_json(out/'status.json',{'status':'RUNNING'})
    try:
        info,meta=inspect_source(paths,topic); save_json(out/'source.json',info)
        # Bound planning from the already-read headers, without a second MCAP scan.
        camera_ns=meta[-1]['stamp_ns']-meta[0]['stamp_ns']
        log_ns=max(m['log_ns'] for m in meta)-min(m['log_ns'] for m in meta)
        if abs(camera_ns-log_ns)>max(3*10**9,log_ns//100):
            raise ValueError('TIMESTAMP_CLOCK_JUMP: camera and MCAP elapsed times differ')
        window_ns=round(window*1e9)
        if window_ns<=0 or (camera_ns+info['median_period_ns']+window_ns-1)//window_ns>len(meta):
            raise ValueError('TIMESTAMP_WINDOW_LIMIT: window count exceeds source frame count')
        windows=plan_windows([m['stamp_ns'] for m in meta],info['median_period_ns'],window,interval)
        selected={i for w in windows for i in w['indices']}; stamps={m['stamp_ns']:m for m in meta}
        (out/'frames').mkdir(); records=[]; decoder=av.CodecContext.create('h264','r')
        decoder.thread_count=threads; decoder.thread_type='SLICE'
        phase=Counter(); decoded=[]; itr=iter(messages(paths,topic)); count=0
        def receive(frames):
            nonlocal count
            for frame in frames:
                count+=1
                if frame.pts is None or frame.pts not in stamps: raise ValueError('Missing/unmapped decoder PTS')
                m=stamps[frame.pts]; idx=m['index']; decoded.append(idx)
                if [frame.width,frame.height]!=info['decoded_dimensions']: raise ValueError('Dimensions changed')
                if idx not in selected: continue
                t=time.perf_counter(); im=frame.to_image(); phase['rgb_conversion_s']+=time.perf_counter()-t
                t=time.perf_counter(); size=target_size(*im.size,longest)
                if im.size!=size: im=im.resize(size,Image.LANCZOS)
                phase['resize_s']+=time.perf_counter()-t
                record={**m,'decoder_pts':frame.pts,'decoder_time_base':str(frame.time_base),
                        'relative_s':(m['stamp_ns']-info['first_stamp_ns'])/1e9,
                        'path':None,'width':size[0],'height':size[1],'jpeg_bytes':0,
                        'raw_rgb_logical_bytes':size[0]*size[1]*3}
                t=time.perf_counter(); data=jpeg_bytes(im,quality=quality); phase['jpeg_encode_s']+=time.perf_counter()-t
                record['jpeg_bytes']=len(data)
                path=out/'frames'/f'{idx:06d}.jpg'
                t=time.perf_counter(); path.write_bytes(data); phase['write_s']+=time.perf_counter()-t
                record['path']=str(path.relative_to(out))
                t=time.perf_counter(); record['sha256']=hashlib.sha256(data).hexdigest()
                phase['jpeg_sha256_s']+=time.perf_counter()-t
                records.append(record)
        while True:
            t=time.perf_counter()
            try: path,schema,ch,msg=next(itr)
            except StopIteration:
                phase['mcap_read_deserialize_s']+=time.perf_counter()-t; break
            stamp,fid,fmt,payload=CDR(msg.data).image(); phase['mcap_read_deserialize_s']+=time.perf_counter()-t
            if fmt!='h264': raise ValueError('Codec changed')
            packet=av.Packet(payload); packet.pts=stamp; packet.time_base=Fraction(1,10**9)
            t=time.perf_counter(); frames=decoder.decode(packet); phase['decode_s']+=time.perf_counter()-t
            receive(frames)
        t=time.perf_counter(); frames=decoder.decode(None); phase['decode_s']+=time.perf_counter()-t; receive(frames)
        if sorted(decoded)!=list(range(len(meta))) or len(records)!=len(selected): raise ValueError('Packet/frame mapping or output coverage mismatch')
        save_json(out/'all_frames.json',meta); save_json(out/'windows.json',windows); save_json(out/'samples.json',records)
        result={'status':'PASS','config':{'topic':topic,'window_s':window,'interval_s':interval,'longest_edge':longest,
                 'quality':quality,'threads':threads,'transport':'jpeg-disk','jpeg_subsampling':2,'resize':'Pillow LANCZOS; preserve ratio; no upscale'},
                'versions':{'python':sys.version,'av':av.__version__,'mcap':mcap.__version__,'pillow':pillow_version,
                            'libav':{k:list(v) for k,v in av.library_versions.items()},'platform':platform.platform()},
                'decoder':{'codec':decoder.name,'thread_count':decoder.thread_count,'thread_type':str(decoder.thread_type)},
                'inspection_s':info['inspection_s'],'phases_s':dict(phase),'decoded_frames':count,
                'sampled_images':len(records),'saved_images':sum(r['path'] is not None for r in records),'windows':len(windows),'output_jpeg_bytes':sum(r['jpeg_bytes'] for r in records),
                'input_bytes':sum(p['bytes'] for p in info['parts']),'input_duration_s':info['coverage_duration_s'],
                'timing_note':'Sequential non-overlapping Python call timers; decoder SLICE threads join per decode call. Total includes inspection, second MCAP pass, selection, metadata and validation; initial interpreter/import startup excluded. Writes use OS page cache, no fsync or cache flushing. Read/deserialization includes chunk decompression, not physical disk alone. No strict cold-start claim.',
                'mapping_note':'MCAP message payload = one H264 access unit, verified exactly one mapped decoded frame per message across whole stream; PTS assigned from original header stamps, all dependency frames decoded, only selected RGB converted/saved.'}
        result['total_wall_s']=time.perf_counter()-total_start
        result['seconds_per_source_minute']=result['total_wall_s']*60/info['coverage_duration_s']
        save_json(out/'timing.json',result); save_json(out/'status.json',{'status':'PASS'})
        return result
    except Exception as e:
        save_json(out/'status.json',{'status':'FAILED','error':repr(e),'wall_s':time.perf_counter()-total_start}); raise

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',required=True,type=Path); p.add_argument('--topic',required=True)
    p.add_argument('--output',required=True,type=Path); p.add_argument('--interval',type=float,default=.5)
    p.add_argument('--longest-edge',type=int,default=960); p.add_argument('--window',type=float,default=3); p.add_argument('--threads',type=int,default=1)
    a=p.parse_args(); paths=sorted(a.source.glob('*.mcap')) if a.source.is_dir() else [a.source]
    if not paths: p.error('No input MCAP')
    print(json.dumps(run(paths,a.topic,a.output,a.interval,a.longest_edge,a.window,a.threads),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
