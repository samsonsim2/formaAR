import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import * as THREE from 'three';
import cvModule from '@techstark/opencv-js';
import './styles.css';

const objects = [
  {id:'chair', name:'Lounge chair', icon:'♜', color:'#ef653b'},
  {id:'lamp', name:'Floor lamp', icon:'♟', color:'#ffd066'},
  {id:'plant', name:'Monstera', icon:'✦', color:'#5f9360'},
];

function useSurfaceTracking(video, debugCanvas, manualPoints, freezeDetection, debugMode, enabled) {
  const [tracking, setTracking] = useState({found:false, quality:0, points:[], quad:[], quadFresh:false, diagnostics:{brightness:0,contrast:0,texture:0}, x:0, y:0, roll:0});
  const anchor = useRef({x:0,y:0,roll:0});
  const resetAnchor = () => { anchor.current={x:0,y:0,roll:0}; };
  useEffect(()=>{
    if(!enabled) return;
    const canvas=document.createElement('canvas'), ctx=canvas.getContext('2d',{willReadFrequently:true});
    const W=1080,H=1920; canvas.width=W;canvas.height=H;
    let cv,previous,features,raf,stable=0,lastUi=0,lastProcess=0,stopped=false,frameNo=0,lastQuad=[],quadMisses=99,manualLocked=false;
    const dispose=()=>{previous?.delete();features?.delete();previous=null;features=null};
    const getCv=async()=>{const mod=cvModule instanceof Promise?await cvModule:cvModule;if(mod.Mat)return mod;await new Promise(resolve=>{mod.onRuntimeInitialized=resolve});return mod};
    const detectPoints=gray=>{const detector=new cv.FastFeatureDetector();detector.setThreshold(14);detector.setNonmaxSuppression(true);const keys=new cv.KeyPointVector();detector.detect(gray,keys);const data=[];for(let i=0;i<Math.min(keys.size(),160);i++){const p=keys.get(i).pt;data.push(p.x,p.y)}detector.delete();keys.delete();return cv.matFromArray(data.length/2,1,cv.CV_32FC2,data)};
    const insideQuad=(x,y,q)=>{let inside=false;for(let i=0,j=q.length-1;i<q.length;j=i++){const a=q[i],b=q[j];if(((a.y>y)!==(b.y>y))&&(x<(b.x-a.x)*(y-a.y)/(b.y-a.y)+a.x))inside=!inside}return inside};
    const tick=now=>{
      if(stopped)return;
      if(video.current?.readyState<2){raf=requestAnimationFrame(tick);return}
      if(now-lastProcess<125){raf=requestAnimationFrame(tick);return}lastProcess=now;
      const vw=video.current.videoWidth,vh=video.current.videoHeight,target=W/H,source=vw/vh;let sx=0,sy=0,sw=vw,sh=vh;if(source>target){sw=vh*target;sx=(vw-sw)/2}else{sh=vw/target;sy=(vh-sh)/2}ctx.drawImage(video.current,sx,sy,sw,sh,0,0,W,H);
      const rgba=cv.imread(canvas),current=new cv.Mat();cv.cvtColor(rgba,current,cv.COLOR_RGBA2GRAY);rgba.delete();
      if(manualPoints.current.length===4){lastQuad=manualPoints.current.map(p=>({...p}));quadMisses=0;manualLocked=true;manualPoints.current=[]}
      if(frameNo++%4===0&&!manualLocked&&!freezeDetection.current){
        const edges=new cv.Mat(),contours=new cv.MatVector(),hierarchy=new cv.Mat();cv.Canny(current,edges,55,145);cv.findContours(edges,contours,hierarchy,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE);
        let best=[],bestArea=0;
        for(let i=0;i<contours.size();i++){const c=contours.get(i),peri=cv.arcLength(c,true),poly=new cv.Mat();cv.approxPolyDP(c,poly,.025*peri,true);const area=Math.abs(cv.contourArea(poly)),minArea=W*H*.0025,maxArea=W*H*.88;if(poly.rows===4&&area>minArea&&area<maxArea&&cv.isContourConvex(poly)&&area>bestArea){bestArea=area;best=Array.from(poly.data32S).reduce((a,v,j)=>{if(j%2===0)a.push({x:v,y:poly.data32S[j+1]});return a},[])}poly.delete();c.delete()}
        if(best.length===4){lastQuad=best;quadMisses=0}else{quadMisses++;if(quadMisses>=2)lastQuad=[]}edges.delete();contours.delete();hierarchy.delete();
      }
      if(!previous){previous=current;features=detectPoints(previous);raf=requestAnimationFrame(tick);return}
      if(!features||features.rows<14){features?.delete();features=detectPoints(previous)}
      const next=new cv.Mat(),status=new cv.Mat(),error=new cv.Mat();
      cv.calcOpticalFlowPyrLK(previous,current,features,next,status,error,new cv.Size(21,21),3,new cv.TermCriteria(cv.TermCriteria_COUNT|cv.TermCriteria_EPS,30,.01));
      const from=[],to=[];for(let i=0;i<status.rows;i++)if(status.data[i]){const x=features.data32F[i*2],y=features.data32F[i*2+1];if(lastQuad.length!==4||(!freezeDetection.current&&quadMisses===0)||insideQuad(x,y,lastQuad)){from.push(x,y);to.push(next.data32F[i*2],next.data32F[i*2+1])}}
      let inliers=from.length,dx=0,dy=0,dr=0,motion=null,homography=null;
      if(from.length>=8){const a=cv.matFromArray(from.length/2,1,cv.CV_32FC2,from),b=cv.matFromArray(to.length/2,1,cv.CV_32FC2,to),mask=new cv.Mat(),h=cv.findHomography(a,b,cv.RANSAC,3,mask);if(!h.empty()){const m=h.data64F.length?h.data64F:h.data32F;homography=Array.from(m);dx=m[2];dy=m[5];dr=Math.atan2(m[3],m[0]);inliers=Array.from(mask.data).filter(Boolean).length}a.delete();b.delete();mask.delete();h.delete()}
      if(!homography&&from.length>=6){const a=cv.matFromArray(from.length/2,1,cv.CV_32FC2,from),b=cv.matFromArray(to.length/2,1,cv.CV_32FC2,to),mask=new cv.Mat();const affine=cv.estimateAffine2D(a,b,mask,cv.RANSAC,2.5);if(!affine.empty()){const m=affine.data64F.length?affine.data64F:affine.data32F;motion=Array.from(m);dx=m[2];dy=m[5];dr=Math.atan2(m[3],m[0]);inliers=Array.from(mask.data).filter(Boolean).length}a.delete();b.delete();mask.delete();affine.delete()}
      const quality=Math.min(1,inliers/32);stable=(quality>.45||lastQuad.length===4)?Math.min(30,stable+1):Math.max(0,stable-2);const found=stable>4;
      if(found){anchor.current.x-=dx/W;anchor.current.y-=dy/H;anchor.current.roll-=dr}
      if((quadMisses>0||manualLocked||freezeDetection.current)&&lastQuad.length===4&&homography)lastQuad=lastQuad.map(p=>{const w=homography[6]*p.x+homography[7]*p.y+homography[8];return{x:(homography[0]*p.x+homography[1]*p.y+homography[2])/w,y:(homography[3]*p.x+homography[4]*p.y+homography[5])/w}});
      else if((quadMisses>0||manualLocked||freezeDetection.current)&&lastQuad.length===4&&motion)lastQuad=lastQuad.map(p=>({x:motion[0]*p.x+motion[1]*p.y+motion[2],y:motion[3]*p.x+motion[4]*p.y+motion[5]}));
      if(manualLocked){quadMisses=homography&&inliers>=8?0:quadMisses+1;if(quadMisses>=8){lastQuad=[];manualLocked=false}}
      if(freezeDetection.current&&!manualLocked){quadMisses=homography&&inliers>=8?0:quadMisses+1;if(quadMisses>=8)lastQuad=[]}
      const points=[];for(let i=0;i<to.length&&points.length<28;i+=2)points.push({x:to[i]/W*160,y:to[i+1]/H*120});
      let sum=0,sumSq=0,textureScore=0,samples=0;for(let i=W+1;i<current.data.length-W-1;i+=64){const v=current.data[i];sum+=v;sumSq+=v*v;textureScore+=Math.abs(v-current.data[i-1])+Math.abs(v-current.data[i-W]);samples++}const brightness=sum/samples,contrast=Math.sqrt(Math.max(0,sumSq/samples-brightness*brightness)),texture=textureScore/(samples*2),diagnostics={brightness:Math.round(brightness),contrast:Math.round(contrast),texture:Math.round(texture)};
      previous.delete();previous=current;features.delete();features=next.clone();next.delete();status.delete();error.delete();
      const out=debugCanvas.current;if(out){if(out.width!==W){out.width=W;out.height=H}if(debugMode.current!=='off'){const view=new cv.Mat();if(debugMode.current==='edges'){const edge=new cv.Mat();cv.Canny(current,edge,55,145);cv.cvtColor(edge,view,cv.COLOR_GRAY2RGBA);edge.delete()}else cv.cvtColor(current,view,cv.COLOR_GRAY2RGBA);cv.imshow(out,view);view.delete()}const dc=out.getContext('2d');if(debugMode.current==='off')dc.clearRect(0,0,W,H);dc.fillStyle='#76e6a5';for(const p of points){dc.beginPath();dc.arc(p.x/160*W,p.y/120*H,5,0,Math.PI*2);dc.fill()}if(lastQuad.length===4){dc.save();dc.globalAlpha=Math.max(.3,1-quadMisses/16);dc.setLineDash(quadMisses?[12,8]:[]);dc.strokeStyle='#ffcc66';dc.lineWidth=7;dc.fillStyle='#ffcc6622';dc.beginPath();lastQuad.forEach((p,i)=>i?dc.lineTo(p.x,p.y):dc.moveTo(p.x,p.y));dc.closePath();dc.fill();dc.stroke();dc.restore()}}
      if(now-lastUi>80){lastUi=now;setTracking({found,quality,points,quad:lastQuad,quadFresh:quadMisses===0,diagnostics,...anchor.current})}
      raf=requestAnimationFrame(tick);
    };
    getCv().then(module=>{if(stopped)return;cv=module;raf=requestAnimationFrame(tick)}).catch(()=>setTracking(t=>({...t,found:false,quality:0})));
    return()=>{stopped=true;cancelAnimationFrame(raf);dispose()};
  },[enabled,video,debugCanvas,manualPoints,freezeDetection,debugMode]);
  return [tracking,resetAnchor];
}

function TrackingPlane({visible,tracking,placed}){
  if(!visible)return null;
  const transform=`translate(${tracking.x*100}vw,${tracking.y*100}vh) rotate(${tracking.roll}rad)`;
  return <div aria-label="Tracked surface" style={{position:'absolute',zIndex:1,left:'50%',top:'60%',width:'min(72vw,330px)',height:'min(28vw,125px)',transform:`translate(-50%,-50%) ${transform} perspective(420px) rotateX(62deg)`,transformOrigin:'50% 50%',border:`2px solid ${placed?'#76e6a5':'#e4bd72'}`,borderRadius:'50%',backgroundImage:`linear-gradient(${placed?'#76e6a566':'#e4bd7266'} 1px,transparent 1px),linear-gradient(90deg,${placed?'#76e6a566':'#e4bd7266'} 1px,transparent 1px)`,backgroundSize:'24px 24px',backgroundPosition:'center',boxShadow:`0 0 18px ${placed?'#76e6a577':'#e4bd7277'},inset 0 0 25px #0008`,transition:'transform 80ms linear,border-color .2s',pointerEvents:'none'}}>
    <span style={{position:'absolute',left:'50%',top:'50%',width:12,height:12,borderRadius:'50%',background:placed?'#76e6a5':'#e4bd72',transform:'translate(-50%,-50%)',boxShadow:'0 0 0 7px #ffffff22,0 0 15px currentColor'}}/>
    <b style={{position:'absolute',left:'50%',top:'100%',transform:'translate(-50%,12px) rotateX(-62deg)',fontSize:8,letterSpacing:'.16em',whiteSpace:'nowrap',color:placed?'#a9f2c5':'#f0d59f'}}>TRACKED SURFACE · {Math.round(tracking.quality*100)}%</b>
  </div>
}

function TrackedBallPlane({quad,placed,mirrored}){
  const mount=useRef(null),quadRef=useRef(quad);quadRef.current=quad;
  useEffect(()=>{
    const el=mount.current,scene=new THREE.Scene(),camera=new THREE.OrthographicCamera(0,el.clientWidth,0,el.clientHeight,-1000,1000);camera.position.z=100;
    const renderer=new THREE.WebGLRenderer({alpha:true,antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(el.clientWidth,el.clientHeight);renderer.domElement.style.display='block';renderer.domElement.style.width='100%';renderer.domElement.style.height='100%';el.appendChild(renderer.domElement);
    const grid=document.createElement('canvas');grid.width=256;grid.height=256;const gc=grid.getContext('2d');gc.strokeStyle='#63a8ff';gc.lineWidth=14;gc.beginPath();gc.arc(128,128,104,0,Math.PI*2);gc.stroke();gc.fillStyle='#63a8ff';gc.beginPath();gc.arc(128,128,12,0,Math.PI*2);gc.fill();
    const texture=new THREE.CanvasTexture(grid),geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(12),3));geometry.setAttribute('uv',new THREE.Float32BufferAttribute([0,0,1,0,1,1,0,1],2));geometry.setIndex([0,1,2,0,2,3]);
    const plane=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({map:texture,transparent:true,alphaTest:.05,side:THREE.DoubleSide,depthWrite:true}));plane.renderOrder=0;scene.add(plane);
    const ball=new THREE.Mesh(new THREE.SphereGeometry(70,40,30),new THREE.MeshStandardMaterial({color:'#287dff',roughness:.3,metalness:.06,depthWrite:true}));ball.position.z=30;ball.renderOrder=2;scene.add(ball);
    const shadow=new THREE.Mesh(new THREE.CircleGeometry(70,40),new THREE.MeshBasicMaterial({color:0x000000,transparent:true,opacity:.32,depthWrite:true}));shadow.scale.y=.35;shadow.position.z=5;shadow.renderOrder=1;scene.add(shadow);
    scene.add(new THREE.HemisphereLight(0xffffff,0x4a2417,2.8));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(-200,-300,400);scene.add(light);
    let raf,start=performance.now();const animate=now=>{raf=requestAnimationFrame(animate);const q=quadRef.current;if(q?.length===4){const s=el.clientWidth/1080,base=q.map(p=>({x:p.x*s,y:p.y*s})),cx=base.reduce((n,p)=>n+p.x,0)/4,cy=base.reduce((n,p)=>n+p.y,0)/4,screen=base.map(p=>({x:cx+(p.x-cx)*2.1,y:cy+(p.y-cy)*2.1})),pos=geometry.attributes.position.array;screen.forEach((p,i)=>{pos[i*3]=p.x;pos[i*3+1]=p.y;pos[i*3+2]=0});geometry.attributes.position.needsUpdate=true;const bob=Math.abs(Math.sin((now-start)/520))*Math.min(160*s,195),visualScale=Math.min(s*2.1,4.6);ball.scale.setScalar(visualScale);shadow.scale.set(1,.35,1).multiplyScalar(visualScale);ball.position.x=cx;ball.position.y=cy-bob-82*s;shadow.position.x=cx;shadow.position.y=cy;ball.rotation.y+=(now-start)*.000003;plane.visible=ball.visible=shadow.visible=true}else plane.visible=ball.visible=shadow.visible=false;renderer.render(scene,camera)};animate(start);
    const resize=()=>{renderer.setSize(el.clientWidth,el.clientHeight);camera.right=el.clientWidth;camera.bottom=el.clientHeight;camera.updateProjectionMatrix()};addEventListener('resize',resize);return()=>{cancelAnimationFrame(raf);removeEventListener('resize',resize);geometry.dispose();texture.dispose();plane.material.dispose();ball.geometry.dispose();ball.material.dispose();shadow.geometry.dispose();shadow.material.dispose();renderer.dispose();el.removeChild(renderer.domElement)};
  },[]);
  return <div ref={mount} style={{position:'absolute',zIndex:3,inset:0,transform:mirrored?'scaleX(-1)':'none',pointerEvents:'none',opacity:placed?1:.58,transition:'opacity .2s'}}/>;
}

function ThreeScene({placed, selected, scale, rotation, tracking}) {
  const mount = useRef(null);
  const model = useRef(null);
  useEffect(() => {
    const el=mount.current, scene=new THREE.Scene();
    const camera=new THREE.PerspectiveCamera(45,el.clientWidth/el.clientHeight,.1,100); camera.position.z=5;
    const renderer=new THREE.WebGLRenderer({alpha:true,antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setSize(el.clientWidth,el.clientHeight); renderer.shadowMap.enabled=true; el.appendChild(renderer.domElement);
    scene.add(new THREE.HemisphereLight(0xffffff,0x402a22,2.5)); const sun=new THREE.DirectionalLight(0xffffff,3); sun.position.set(3,5,4); scene.add(sun);
    const floor=new THREE.Mesh(new THREE.CircleGeometry(1.15,64),new THREE.ShadowMaterial({opacity:.18})); floor.rotation.x=-Math.PI/2; floor.position.y=-1.15; floor.receiveShadow=true; scene.add(floor);
    const group=new THREE.Group(); model.current=group; scene.add(group);
    const mat=(c)=>new THREE.MeshStandardMaterial({color:c,roughness:.68,metalness:.05});
    const add=(g,p,m)=>{const x=new THREE.Mesh(g,m); Object.assign(x.position,p); x.castShadow=true; group.add(x); return x};
    if(selected.id==='chair'){
      add(new THREE.BoxGeometry(2.05,.28,1.55),{x:0,y:-.34,z:0},mat(selected.color));
      add(new THREE.BoxGeometry(2.05,1.18,.25),{x:0,y:.3,z:-.65},mat('#d95532'));
      [-.83,.83].forEach(x=>[-.48,.52].forEach(z=>{const leg=add(new THREE.CylinderGeometry(.08,.1,.85,16),{x,y:-.9,z},mat('#512f24')); leg.rotation.z=x<0?-.05:.05}));
    } else if(selected.id==='lamp'){
      add(new THREE.CylinderGeometry(.52,.7,.15,32),{x:0,y:-1.05,z:0},mat('#3d342c'));
      add(new THREE.CylinderGeometry(.055,.07,2.2,20),{x:0,y:.02,z:0},mat('#6d5c50'));
      const shade=add(new THREE.CylinderGeometry(.48,.8,.85,32,1,true),{x:0,y:1.05,z:0},mat(selected.color)); shade.material.side=THREE.DoubleSide;
    } else {
      add(new THREE.CylinderGeometry(.58,.42,.82,32),{x:0,y:-.76,z:0},mat('#d96645'));
      for(let i=0;i<9;i++){const leaf=add(new THREE.SphereGeometry(.3,20,16),{x:Math.sin(i*2.1)*.48,y:-.05+(i%3)*.34,z:Math.cos(i*2.1)*.35},mat(i%2?'#487a4b':'#6ba76b')); leaf.scale.set(.55,1.55,.28); leaf.rotation.z=Math.sin(i)*.75;}
    }
    group.visible=placed; let frame; const animate=()=>{frame=requestAnimationFrame(animate); renderer.render(scene,camera)}; animate();
    const resize=()=>{camera.aspect=el.clientWidth/el.clientHeight;camera.updateProjectionMatrix();renderer.setSize(el.clientWidth,el.clientHeight)}; addEventListener('resize',resize);
    return()=>{cancelAnimationFrame(frame);removeEventListener('resize',resize);renderer.dispose();el.removeChild(renderer.domElement)};
  },[selected,placed]);
  useEffect(()=>{if(model.current){model.current.scale.setScalar(scale);model.current.rotation.y=rotation}},[scale,rotation]);
  return <div className="three" ref={mount} style={{transform:`translate(${tracking.x*100}vw,${tracking.y*100}vh) rotate(${tracking.roll}rad)`,transition:'transform 80ms linear',transformOrigin:'50% 72%'}}/>;
}

function App(){
 const video=useRef(null),debugCanvas=useRef(null),manualPoints=useRef([]),freezeDetection=useRef(false),debugModeRef=useRef('off'),[debugMode,setDebugMode]=useState('off'),[placed,setPlaced]=useState(false),[started,setStarted]=useState(false),[cameraError,setCameraError]=useState(''),[cameraResolution,setCameraResolution]=useState(''),[mirrored,setMirrored]=useState(false);
 freezeDetection.current=placed;debugModeRef.current=debugMode;
 const [tracking]=useSurfaceTracking(video,debugCanvas,manualPoints,freezeDetection,debugModeRef,started&&!cameraError), detected=tracking.found;
 const pointInPlane=(x,y,q)=>{let inside=false;for(let i=0,j=q.length-1;i<q.length;j=i++){const a=q[i],b=q[j];if(((a.y>y)!==(b.y>y))&&(x<(b.x-a.x)*(y-a.y)/(b.y-a.y)+a.x))inside=!inside}return inside};
 const start=async()=>{try{const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:3840},height:{ideal:2160},frameRate:{ideal:30}},audio:false});video.current.srcObject=stream;await video.current.play();const s=stream.getVideoTracks()[0].getSettings();setMirrored(s.facingMode==='user');setCameraResolution(`${s.width||'?'} × ${s.height||'?'}`);setStarted(true)}catch(e){setCameraError('Camera access is needed to scan your space.');setStarted(true)}};
 useEffect(()=>()=>{video.current?.srcObject?.getTracks().forEach(t=>t.stop())},[]);
 useEffect(()=>{if(placed&&tracking.quad.length!==4)setPlaced(false)},[placed,tracking.quad.length]);
 return <div style={{width:'100vw',height:'100dvh',display:'grid',placeItems:'center',background:'#0d0c0a'}}><main style={{width:'min(100vw,calc(100dvh * 9 / 16))',height:'min(100dvh,calc(100vw * 16 / 9))',aspectRatio:'9 / 16',margin:0,boxShadow:'0 0 70px #000'}}>
<video ref={video} className={`camera-feed${mirrored?' camera-feed--mirrored':''}`} muted playsInline/>
   <div className="wash"/>
   <header><button className="round">×</button><div className="brand"><span>FORMA</span><b>SPACE SCAN</b></div><button className="round">•••</button></header>
   {!started ? <section className="permission"><div className="cube">◇</div><p className="eyebrow">OPENCV DIAGNOSTIC</p><h1>Find a<br/><em>flat face.</em></h1><p>Aim at a box, book, tabletop, wall, or any visible planar surface.</p><button className="primary" onClick={start}>Start camera <span>→</span></button><small>All processing stays in this browser.</small></section> : <>
    <canvas ref={debugCanvas} style={{position:'absolute',zIndex:2,inset:0,width:'100%',height:'100%',opacity:debugMode!=='off'?1:0,pointerEvents:'none',transform:mirrored?'scaleX(-1)':'none'}}/>
    <TrackedBallPlane quad={tracking.quad} placed={placed} mirrored={mirrored}/>
    <div onClick={e=>{const r=e.currentTarget.getBoundingClientRect(),screenX=(e.clientX-r.left)/r.width,x=(mirrored?1-screenX:screenX)*1080,y=(e.clientY-r.top)/r.height*1920;if(tracking.quad.length===4&&pointInPlane(x,y,tracking.quad))setPlaced(true)}} style={{position:'absolute',zIndex:4,inset:0,cursor:tracking.quad.length===4?'pointer':'default',WebkitTapHighlightColor:'transparent',touchAction:'manipulation',userSelect:'none'}}/>
    <button onClick={()=>setDebugMode(v=>v==='off'?'gray':v==='gray'?'edges':'off')} style={{position:'absolute',zIndex:8,right:18,top:82,padding:'9px 12px',border:'1px solid #ffffff55',borderRadius:3,background:debugMode!=='off'?'#76e6a5':'#17140fcc',color:debugMode!=='off'?'#102016':'#fff',fontSize:9,letterSpacing:'.13em'}}>{debugMode==='off'?'DEBUG VIEW':debugMode==='gray'?'GRAYSCALE':'EDGE MAP'}</button>
    <div className="status"><span className={tracking.quad.length===4?'dot green':'dot'}/>{cameraError?'CAMERA UNAVAILABLE':placed?'SURFACE LOCKED':tracking.quad.length===4?'SURFACE READY':'SEARCHING FOR A SURFACE'}<small>{cameraError?'Allow camera access and reload':placed?'Tracking plane orientation':tracking.quad.length===4?'Tap the plane to place it':'Move slowly around a visible flat face'}</small></div>
    <div style={{position:'absolute',zIndex:6,left:18,right:18,bottom:24,padding:'13px 15px',background:'#111c',border:'1px solid #ffffff25',fontSize:10,lineHeight:1.6,letterSpacing:'.08em',pointerEvents:'none'}}><b style={{color:placed?'#76e6a5':'#ffcc66'}}>{placed?'● PLACED — TRACKING ACTIVE':'● PREVIEW — TAP TO PLACE'}</b><br/><span style={{color:'#aaa'}}>{debugMode!=='off'?`${tracking.points.length} features · ${tracking.quad.length===4?'plane candidate':'no quadrilateral'} · light ${tracking.diagnostics.brightness}/255 · contrast ${tracking.diagnostics.contrast} · texture ${tracking.diagnostics.texture}`:'9:16 crop · CV 1080 × 1920 · 1s tracking-loss reset'}</span></div>
   </>}
 </main></div>
}
createRoot(document.getElementById('root')).render(<App/>);
