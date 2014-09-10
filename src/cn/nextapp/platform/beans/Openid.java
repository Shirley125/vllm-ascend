package cn.nextapp.platform.beans;

import java.sql.Timestamp;
import java.util.List;
import java.util.HashMap;

import my.db.QueryHelper;

/**
 * OpenID帐号
 * @author Winter Lau
 * @date 2011-12-28 下午12:10:07
 */
public class Openid extends Pojo {
	
	public final static HashMap<String, Byte> TYPES = new HashMap<String, Byte>(){{
		put("weibo", 	(byte)1);
		put("qq", 		(byte)2);
		put("google", 	(byte)3);
		put("yahoo", 	(byte)4);
		put("hotmail", 	(byte)5);
		put("facebook", (byte)6);
		put("twitter", 	(byte)7);
	}};
	
	/**
	 * 列出用户绑定的所有登录方式
	 * @param user
	 * @return
	 */
	public static List<Openid> listOfUser(int user) {
		String sql = "SELECT * FROM nap_openids WHERE user = ?";
		return QueryHelper.query(Openid.class, sql, user);
	}
	
	/**
	 * 执行 OpenID 登录
	 * @param openid
	 * @param provider
	 * @return
	 */
	public static Openid loginViaOpenId(String openid, String provider) {
		String sql = "SELECT user FROM nap_openids WHERE openid = ? AND type = ?";
		Openid user = QueryHelper.read(Openid.class, sql, openid, _GetProviderType(provider));
		if(user != null){
			sql = "UPDATE nap_openids SET last_login = ? WHERE id = ?";
			QueryHelper.update(sql, new Timestamp(System.currentTimeMillis()), user.getId());
		}
		return user;
	}
	
	public static Openid getOpenid(String openid, String provider) {
		String sql = "SELECT user FROM nap_openids WHERE openid = ? AND type = ?";
		return QueryHelper.read(Openid.class, sql, openid, _GetProviderType(provider));
	}
	
	private static byte _GetProviderType(String pvd) {
		Byte type = TYPES.get(pvd);
		return (type!=null)?type:0;
	}

	/**
	 * 生成新的 OpenID 记录
	 * @param user
	 * @param openid
	 * @param provider
	 * @param name
	 */
	public static int saveOpenId(int user, String openid, String provider, String name) {
		Openid id = new Openid();
		id.setUser(user);
		Timestamp current_time = new Timestamp(System.currentTimeMillis());
		id.setCreate_time(current_time);
		id.setLast_login(current_time);
		id.setOpenid(openid);
		id.setType(_GetProviderType(provider));
		id.setName(name);
		return id.Save();
	}	

	@Override
	protected String insertSQLExtend() {
		return "ON DUPLICATE KEY UPDATE last_login = CURRENT_TIMESTAMP";
	}
	
	private int user;
	private String openid;
	private byte type;
	private String name;
	private Timestamp create_time;
	private Timestamp last_login;

	@Override
	protected boolean IsObjectCachedByID() {
		return true;
	}
	
	public int getUser() { return user; }
	public void setUser(int user) { this.user = user; }
	public String getOpenid() { return openid; }
	public void setOpenid(String openid) { this.openid = openid; }
	public byte getType() { return type; }
	public void setType(byte type) { this.type = type; }
	public String getName() { return name; }
	public void setName(String name) { this.name = name; }
	public Timestamp getCreate_time() { return create_time; }
	public void setCreate_time(Timestamp create_time) { this.create_time = create_time; }
	public Timestamp getLast_login() { return last_login; }
	public void setLast_login(Timestamp last_login) { this.last_login = last_login; }
	
	public String typeDesc() {
		for(String typeDesc : TYPES.keySet()){
			if(TYPES.get(typeDesc) == this.type)
				return typeDesc;
		}
		return null;
	}
}
