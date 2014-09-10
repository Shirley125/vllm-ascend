package cn.nextapp.platform.beans;

import java.util.List;
import java.sql.Timestamp;

import cn.nextapp.platform.SmtpHelper;
import cn.nextapp.platform.toolbox.LinkTool;

import my.db.QueryHelper;

/**
 * App版本信息
 * @author Winter Lau
 * @date 2011-12-29 下午11:18:00
 */
public class Version extends Pojo {
	
	public final static Version INSTANCE = new Version();

	public String url(){
		return LinkTool.action("dl?app=" + App().getGuid() + "&ver="+this.version);
	}
	
	public static List<Version> listVersions(int app){
		String sql = "SELECT * FROM nap_versions WHERE app = ? ORDER BY id DESC";
		return QueryHelper.query(Version.class, sql, app);
	}
	
	/**
	 * 检查新版本
	 * @param repo_id
	 * @return
	 */
	public static Version checkNewVersion(int current_id) {
		Version ver = INSTANCE.Get(current_id);
		if(ver == null)
			return null;
		String sql = "SELECT * FROM nap_versions WHERE app = ? AND client_type = ? AND id > ? ORDER BY id DESC LIMIT 1";
		return QueryHelper.read_cache(Version.class, INSTANCE.CacheRegion(), "NEW"+current_id, sql, ver.getApp(), ver.getClient_type(), ver.getId());
	}
	
	public void incDlCount(int c) {
		String sql = "UPDATE osc_versions SET dl_count = dl_count + ? WHERE id = ?";
		QueryHelper.update(sql, c, getId());
	}
	
	/**
	 * 使用最新的版本生成初始化数据
	 * @param app
	 */
	public static void init(int app) {
		for(Repository repo : Repository.ListNewestVersions()){
			Version v = new Version();
			v.setApp(app);
			v.setVersion(repo.getId());
			v.setClient_type(repo.getClient_type());
			v.setBuild_status(App.BUILD_STATUS_NEED_BUILD);
			v.Save();
		}
	}
	
	/**
	 * 开始构建
	 */
	public void beginBuild() {
		String sql = "UPDATE nap_versions SET build_status = ? , build_begin_time = ? WHERE id = ?";
		QueryHelper.update(sql, App.BUILD_STATUS_BUILDING, new Timestamp(System.currentTimeMillis()), getId());
	}
	
	/**
	 * 构建结束
	 * @param success
	 * @param app_path
	 * @throws Exception 
	 */
	public void endBuild(boolean success, String app_path) {
		String sql = "UPDATE nap_versions SET app_path = ?, build_status = ? , build_end_time = ? WHERE id = ?";
		QueryHelper.update(sql, app_path, (success?App.BUILD_STATUS_DONE:App.BUILD_STATUS_FAILED), new Timestamp(System.currentTimeMillis()), getId());	
		//发送邮件通知
		App app = new App().Get(this.getApp());
		User user = new User().Get(app.getUser());
		try{
			SmtpHelper.sendBuildEndNotifyMail(user, app, this, success);
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	@Override
	protected boolean IsObjectCachedByID() {
		return true;
	}
	
	public App App() {
		return App.INSTANCE.Get(app);
	}

	private int app;
	private int version;
	private byte client_type;
	private String app_path;
	private int dl_count;
	private byte build_status;
	private int error_code;
	private Timestamp build_begin_time;
	private Timestamp build_end_time;
	private Timestamp create_time;
	
	public int getApp() { return app; }
	public void setApp(int app) { this.app = app; }
	public int getVersion() { return version; }
	public void setVersion(int version) { this.version = version; }
	public byte getClient_type() { return client_type; }
	public void setClient_type(byte client_type) { this.client_type = client_type; }
	public String getApp_path() { return app_path; }
	public void setApp_path(String app_path) { this.app_path = app_path; }
	public int getDl_count() { return dl_count; }
	public void setDl_count(int dl_count) { this.dl_count = dl_count; }
	public byte getBuild_status() { return build_status; }
	public void setBuild_status(byte build_status) { this.build_status = build_status; }
	public int getError_code() { return error_code; }
	public void setError_code(int error_code) { this.error_code = error_code; }
	public Timestamp getBuild_begin_time() { return build_begin_time; }
	public void setBuild_begin_time(Timestamp build_begin_time) { this.build_begin_time = build_begin_time; }
	public Timestamp getBuild_end_time() { return build_end_time; }
	public void setBuild_end_time(Timestamp build_end_time) { this.build_end_time = build_end_time; }
	public Timestamp getCreate_time() { return create_time; }
	public void setCreate_time(Timestamp create_time) { this.create_time = create_time; }
	
}
