package cn.nextapp.platform.beans;

import java.sql.Date;
import java.sql.Timestamp;

import java.util.List;

import my.db.QueryHelper;

/**
 * 源码版本库
 * @author Winter Lau
 * @date 2011-12-30 下午3:41:40
 */
public class Repository extends Pojo {

	/**
	 * 列出每个平台最新的版本
	 * @return
	 */
	public static List<Repository> ListNewestVersions() {
		String sql = "SELECT client_type,MAX(id) AS id FROM nap_repositories GROUP BY client_type";
		return QueryHelper.query(Repository.class, sql);
	}

	@Override
	protected boolean IsObjectCachedByID() {
		return true;
	}

	private byte client_type;
	private String version;
	private String src;
	private Date pub_date;
	private Timestamp create_time;
	private String changelog;
	
	public byte getClient_type() {
		return client_type;
	}
	public void setClient_type(byte client_type) {
		this.client_type = client_type;
	}
	public String getVersion() {
		return version;
	}
	public void setVersion(String version) {
		this.version = version;
	}
	public String getSrc() {
		return src;
	}
	public void setSrc(String src) {
		this.src = src;
	}
	public Date getPub_date() {
		return pub_date;
	}
	public void setPub_date(Date pub_date) {
		this.pub_date = pub_date;
	}
	public Timestamp getCreate_time() {
		return create_time;
	}
	public void setCreate_time(Timestamp create_time) {
		this.create_time = create_time;
	}
	public String getChangelog() {
		return changelog;
	}
	public void setChangelog(String changelog) {
		this.changelog = changelog;
	}
	
}
