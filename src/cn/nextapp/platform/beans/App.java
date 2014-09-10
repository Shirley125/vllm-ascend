package cn.nextapp.platform.beans;

import java.util.Date;
import java.util.List;
import java.util.UUID;

import org.apache.commons.lang.StringUtils;

import my.db.POJO;
import my.db.QueryHelper;

/**
 * 网站
 * @author Winter Lau
 */
public class App extends POJO {

	public final static App INSTANCE = new App();

	public final static byte TYPE_BLOG	= 0x01;
	public final static byte TYPE_BBS	= 0x02; 
	
	public final static byte STATUS_NEED_PIC 		= 0x00;	//尚未上传图片
	public final static byte STATUS_NEED_AUDIT 		= 0x01;	//待审批
	public final static byte STATUS_AUDIT_PASSED 	= 0x02;	//审批通过
	public final static byte STATUS_AUDIT_FAILED 	= 0x03;	//审批不通过
	public final static byte STATUS_DONE		 	= 0x04;	//构建完成
	
	public final static byte BUILD_STATUS_NEED_BUILD	= 0x00;	//等待生成
	public final static byte BUILD_STATUS_BUILDING		= 0x01;	//生成中
	public final static byte BUILD_STATUS_DONE			= 0x02;	//生成完毕
	public final static byte BUILD_STATUS_FAILED		= 0x03;	//生成失败

	public static int CountByStatus(int status) {
		String sql = "SELECT COUNT(id) FROM nap_apps WHERE status = ?";
		return (int)QueryHelper.stat(sql, status);
	}
	
	public static List<App> ListByStatus(int status, int page, int count) {
		String sql = "SELECT * FROM nap_apps WHERE status = ?";
		return QueryHelper.query_slice(App.class, sql, page, count, status);
	}
	
	public boolean approve() {
		String sql = "UPDATE nap_versions SET build_status = ? WHERE app = ?";
		QueryHelper.update(sql, App.BUILD_STATUS_NEED_BUILD, getId());
		sql = "UPDATE nap_apps SET status = ? WHERE id = ? AND status = ?";
		return QueryHelper.update(sql, STATUS_AUDIT_PASSED, getId(), STATUS_NEED_AUDIT)>0;
	}

	public boolean deny(String reason) {
		int error_code = Error.auditError(reason).getId();
		String sql = "UPDATE nap_apps SET status = ?, error_code=? WHERE id = ? AND status = ?";
		return QueryHelper.update(sql, STATUS_AUDIT_FAILED, error_code, getId(), STATUS_NEED_AUDIT)>0;
	}
	
	@Override
	public boolean Delete() {
		String sql = "DELETE FROM nap_versions WHERE app = ?";
		QueryHelper.update(sql, getId());
		return super.Delete();
	}

	/**
	 * 根据GUID获取app
	 * @param guid
	 * @return
	 */
	public static App getAppByGuid(String guid) {
		if(StringUtils.isBlank(guid))
			return null;
		String sql = "SELECT * FROM nap_apps WHERE guid = ?";
		return QueryHelper.read(App.class, sql, guid);
	}
	
	public static App getAppByDomain(String domain){
		String sql = "SELECT * FROM nap_apps WHERE domain = ?";
		return QueryHelper.read(App.class, sql, domain);
	}

	public static App getAppByName(String name){
		String sql = "SELECT * FROM nap_apps WHERE name = ?";
		return QueryHelper.read(App.class, sql, name);
	}
	
	/**
	 * 列出某个帐号的所有App
	 * @param user
	 * @return
	 */
	public static List<App> listOfUser(int user){
		String sql = "SELECT * FROM nap_apps WHERE user = ?";
		return QueryHelper.query(App.class, sql, user);
	}
	
	/**
	 * 列出所有的版本
	 * @return
	 */
	public List<Version> versions(){
		return Version.listVersions(getId());
	}
	
	@Override
	public int Save() {
		this.guid = UUID.randomUUID().toString();
		this.status = STATUS_NEED_PIC;
		int app = super.Save();
		return app;
	}

	/**
	 * 更新app的图片信息
	 * @return
	 */
	public boolean Update() {
		String sql = "UPDATE nap_apps SET status=?,name=?,outline=?,home_url=?,plugin_url=?,ico=?,logo=?,welcome=? WHERE id = ?";
		return Evict(QueryHelper.update(sql, STATUS_NEED_AUDIT, name,outline,home_url,plugin_url,ico, logo, welcome, getId())==1);
	}

	/**
	 * 更新app的图片信息
	 * @return
	 */
	public boolean UploadPics() {
		String sql = "UPDATE nap_apps SET status=?,ico=?,logo=?,welcome=? WHERE id = ?";
		return Evict(QueryHelper.update(sql, STATUS_NEED_AUDIT, ico, logo, welcome, getId())==1);
	}
	
	public String GetErrorMsg(){
		return Error.getErrorMsg(error_code);
	}

	@Override
	protected boolean IsObjectCachedByID() {
		return true;
	}

	private int user;
	private String site;	//网站名称，例如“麦金电商”
	private String name;	//app名称，用于在应用列表中的显示，例如“麦金博客”
	private String outline;	//网站的简短说明
	private byte type;		//网站类型：1,blog; 2,bbs
	private String ico;		//网站图标
	private String logo;	//网站Logo
	private String welcome;	//启动屏
	private String style;	//app风格
	private String guid;	//产品唯一编码，目前用于WP7版本
	private String domain;	//网站域名
	private String home_url;//网站首页地址
	private String plugin_url;	//NextApp插件地址
	private Date create_time;	//创建时间
	private byte status;
	private int error_code;
	
	public int getUser() { return user; }
	public void setUser(int user) { this.user = user; }
	public String getSite() { return site; }
	public void setSite(String site) { this.site = site; }
	public String getName() { return name; }
	public void setName(String name) { this.name = name; }
	public String getOutline() { return outline; }
	public void setOutline(String outline) { this.outline = outline; }
	public byte getType() { return type; }
	public void setType(byte type) { this.type = type; }
	public String getIco() { return ico; }
	public void setIco(String ico) { this.ico = ico; }
	public String getLogo() { return logo; }
	public void setLogo(String logo) { this.logo = logo; }
	public String getStyle() { return style; }
	public void setStyle(String style) { this.style = style; }
	public String getHome_url() { return home_url; }
	public void setHome_url(String home_url) { this.home_url = home_url; }
	public String getDomain() { return domain; }
	public void setDomain(String domain) { this.domain = domain; }
	public String getPlugin_url() { return plugin_url; }
	public void setPlugin_url(String plugin_url) { this.plugin_url = plugin_url; }
	public Date getCreate_time() { return create_time; }
	public void setCreate_time(Date create_time) { this.create_time = create_time; }
	public String getWelcome() { return welcome; }
	public void setWelcome(String welcome) { this.welcome = welcome; }
	public String getGuid() { return guid; }
	public void setGuid(String guid) { this.guid = guid; }
	public byte getStatus() { return status; }
	public void setStatus(byte status) { this.status = status; }
	public int getError_code() { return error_code; }
	public void setError_code(int error_code) { this.error_code = error_code; }
}
