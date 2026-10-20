"""Finalize a reviewed local commit and write a truthful local upload receipt."""
from pathlib import Path
import sys,json,csv,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.runtime import read,atomic_json,sha
from prepare_github_upload import audit,staged
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf-8').strip()
def finalize():
    original=list(csv.DictReader((ROOT/'reports/source_provenance.csv').open(encoding='utf-8-sig')))
    assert len(original)==19
    for r in original:
        assert sha(ROOT.parent/r['source'])==r['source_sha256']
        assert sha(ROOT/r['destination'])==r['destination_sha256']
    formal=list(csv.DictReader((ROOT.parent/'reacher_anchor_calibration/results/final_comparison.csv').open(encoding='utf-8-sig')))
    core=list(csv.DictReader((ROOT/'reports/model_manifest.csv').open(encoding='utf-8-sig')))
    for r in core:assert sha(ROOT/r['path'])==r['sha256']
    for j in read(ROOT.parent/'reacher_anchor_calibration/configs/final_freeze.json')['jobs']:
        if j['path']:assert sha(j['path'])==j['sha256']
    assert read(ROOT/'tests/github_isolated_candidate_validation.json')['passed']
    changes=dict(scope='GitHub private packaging; no algorithm or frozen-result changes',
        private_hosting_authorized_by_user=True,public_hosting_authorized=False,
        source_verified_files_unchanged=19,core_models_hash_verified=15,all_frozen_declarations_hash_verified=46,
        io_fixes=['From-scratch anchoring uses template config and newly trained teacher/data, without historical model initialization','Missing-model initialization error identifies required relative path and SHA','Expert-only diagnostic evaluation/recording does not require learned models','README points only to staged resources; unavailable GIF/model links are plain local filenames'],
        original_training_loss_and_optimizer_unchanged=True,git_authentication='Git Credential Manager; no token persisted by project scripts')
    atomic_json(ROOT/'tests/github_packaging_changes.json',changes)
    subprocess.run(['git','add','.'],cwd=ROOT,check=True)
    audit();subprocess.run(['git','add','.'],cwd=ROOT,check=True)
    with (ROOT/'reports/public_file_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','sha256','bytes','staged_git_blob_sha1']);writer.writeheader()
        for p in staged():
            if p=='reports/public_file_manifest.csv':continue
            writer.writerow(dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size,staged_git_blob_sha1=git('rev-parse',':'+p)))
    subprocess.run(['git','add','reports/public_file_manifest.csv'],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    exists=subprocess.run(['git','rev-parse','--verify','HEAD'],cwd=ROOT,capture_output=True).returncode==0
    changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode!=0
    if not exists or changed:subprocess.run(['git','commit','-m','Prepare reproducible Reacher RL portfolio'],cwd=ROOT,check=True)
    write_report()
def write_report():
    state=read(ROOT/'reports/github_upload_runtime.json') if (ROOT/'reports/github_upload_runtime.json').exists() else {}
    head=git('rev-parse','HEAD');files=git('ls-files','-z').split('\0');files=[x for x in files if x]
    success=state.get('remote_verified') and state.get('head')==head
    status='已经成功上传并远程验证' if success else '本地已提交、推送尚未完成：GitHub本机认证/连接待完成'
    checks=read(ROOT/'tests/github_preupload_audit.json');isolated=read(ROOT/'tests/github_isolated_candidate_validation.json')
    origins=git('remote','-v') or '尚未配置；等待认证后确认目标仓库再添加origin'
    url=state.get('url','未创建或未通过认证确认，不能提供已上传仓库URL')
    body=f'''# GitHub上传与发布检查报告

**状态：{status}。**

本报告是本地核验回执，Git忽略该文件，以便准确记录包含最终提交本身的HEAD，避免把报告自己的SHA写入同一提交的循环。报告不包含Token、密码或私钥。

| 项目 | 实际状态 |
|---|---|
| GitHub插件当前账号 | lllppp1022-cmd（插件已实际返回login及id） |
| 插件安装账户/可访问仓库 | 初次检查均空；未暴露创建仓库工具；不能代替git push凭据 |
| 仓库URL | {url} |
| 目标仓库名 | reacher-rl-from-demonstrations |
| 可见性 | {'Private，远程API已验证' if success else '要求Private；远程尚未验证，未改为Public'} |
| 默认分支 | {'main（远程已确认）' if success else '本地main；远程默认分支待验证'} |
| 本地最新commit | {head} |
| 本地提交文件数量 | {len(files)} |
| 已上传文件数量 | {state.get('uploaded_files',0) if success else '未确认，不能用本地文件数冒充'} |
| 是否真正推送完成 | {'是' if success else '否'} |
| 远程HEAD是否与本地一致 | {state.get('remote_head','未验证') if success else '未验证'} |
| 是否允许改为Public | 否；必须另行取得用户确认和适用发布权限 |

## 托管权限和排除范围

用户已在本次任务中明确确认：有权将源码（含src/verified19个文件）、文档、脱敏指标与7张PNG保存到本人GitHub私有仓库。没有授予Public发布或开放源码许可证。所有模型ZIP/PT、专家/诊断NPZ、训练状态/RGB/轨迹、5个GIF和本机环境缓存继续排除。third_party许可证及NOTICE保持。未擅自删除必要源码以绕过权属要求。

实际扫描{checks['candidate_files']}个候选文件，最大{checks['max_file_bytes']}字节；密钥/Token常见格式、本机身份路径、禁止二进制和README失效链接扫描无发现。扫描是启发式检查，不能保证识别全部凭据形式。7张PNG逐一解码，无EXIF，用户已授权Private托管。

5个GIF本地大小约19.65/4.48/7.88/3.12/4.16MB，均未上传，原因是托管权限未确认，并非伪称超过GitHub限制。GitHub官方文档说明普通Git单文件50MiB警告、100MiB禁止，浏览器上传25MiB；当前拟上传文件均远小于这些阈值：[官方大小说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。README保留实际控制截图、流程图和结果图，所有相对链接均指向Git内文件；GIF只列本地文件名。

## 轻量工程核验

- 使用现有Windows Python3.11.17及固定依赖；没有声称全新机器安装或跨平台验证。
- 主入口--help、全部主要模块导入、15项模型注册和本地已授权资产init通过。
- 仅包含暂存文件的隔离目录用python -I测试，不带历史工程、预训练模型或诊断NPZ。指标命令、2回合专家诊断、从头BC/DAgger/Anchoring PPO烟雾流程实际通过；后者1024PPO步、500训练专家标签+250验证标签+250DAgger标签/在线步，非正式研究。
- 无模型时init按预期报错，明确提示BC模型相对路径及SHA，而非伪称已初始化。
- 修正默认烟雾流程误依赖历史资产的I/O问题，优化器/奖励/约束损失保持原实现；19个迁移核心文件、15个核心权重和46个冻结声明哈希复核通过。
- 六条隔离命令结果详见tests/github_isolated_candidate_validation.json；上传前检查见tests/github_preupload_audit.json。
- README数字58.67%/2.3029cm、32.33%/2.7527cm、64.67%/1.9986cm保持真实CSV结果；3训练种子、每模型100冻结初态、100352新增在线步。仍声明单任务仿真、描述性比较，非统计显著或实机部署。

## Git状态与远程操作

```text
{origins}
```

本地提交使用账号对应GitHub noreply邮箱，仅设置本仓库，未修改全局身份。没有force push、删除历史或添加未经授权的项目许可证。

{'仓库存在、Private、main、最新commit、所有文件Git blob哈希、README内容和图片、无受限二进制均已远程实际验证。' if success else 'GitHub插件的目标get_repo返回404，但在插件缺少安装权限时，不能据此断言同名仓库绝对不存在。认证完成后将再次检查，存在历史则不覆盖。本地Git Credential Manager初次无已登录账户，gh未安装；Codex受限终端请求曾命中拒绝连接代理，提升权限的设备登录也未取得验证码并已停止。因此尚未创建、push或伪称远程验收通过。'}

{'待用户另行确认是否Public；当前保持Private。' if success else '下一步需在用户自己的PowerShell完成 git credential-manager github login --username lllppp1022-cmd --device 的官方网页验证，随后继续创建Private空仓库、添加origin、git push -u origin main并读取远程核对。不要把Token或密码发到聊天。'}
'''
    (ROOT/'docs/GitHub上传与发布检查报告.md').write_text(body,encoding='utf-8')
    print(json.dumps(dict(status=status,head=head,local_files=len(files),pushed=bool(success)),ensure_ascii=True))
if __name__=='__main__':
    if '--report-only' in sys.argv:write_report()
    else:finalize()
