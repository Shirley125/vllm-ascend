/**
 * 
 */
package cn.nextapp.platform.beans;

import java.util.Date;
import java.util.List;

import my.db.QueryHelper;

/**
 * 帐号
 * @author Winter Lau
 */
public class User extends Pojo {

	public final static transient byte STATUS_NORMAL = 0x01;
	
	/**
	 * TODO: 判断当前用户是否为管理员
	 * @return
	 */
	public boolean IsAdmin() {
		return true;
	}
	
	/**
	 * 列出该用户所有绑定的登录方式
	 * @return
	 */
	public List<Openid> openids() {
		return Openid.listOfUser(getId());
	}
	
	/**
	 * 列出该用户创建的所有app
	 * @return
	 */
	public List<App> apps() {
		return App.listOfUser(getId());
	}
	
	/**
	 * 添加新用户
	 * @param name
	 * @param email
	 * @return
	 */
	public static User newUser(String name, String email) {
		User user = new User();
		user.setName(name);
		user.setEmail(email);
		user.setStatus(STATUS_NORMAL);
		user.setValidation("10000000");
		user.Save();
		return user;
	}
	
	/**
	 * 更新用户资料
	 * @return
	 */
	public boolean Update() {
		String sql = "UPDATE nap_users SET name=?,email=?,phone=? WHERE id=?";
		return QueryHelper.update(sql, name,email,phone,getId())==1;
	}
	
	private String name;
	private String email;
	private String phone;
	private String validation;
	private byte status;
	private Date create_time;
	
	@Override
	protected boolean IsObjectCachedByID() {
		return true;
	}
	
	@Override
	public int getId() { return super.getId(); }
	public String getName() { return name; }
	public void setName(String name) { this.name = name; }
	public String getEmail() { return email; }
	public void setEmail(String email) { this.email = email; }
	public String getPhone() { return phone; }
	public void setPhone(String phone) { this.phone = phone; }
	public String getValidation() { return validation; }
	public void setValidation(String validation) { this.validation = validation; }
	public byte getStatus() { return status; }
	public void setStatus(byte status) { this.status = status; }
	public Date getCreate_time() { return create_time; }
	public void setCreate_time(Date create_time) { this.create_time = create_time; }

}
