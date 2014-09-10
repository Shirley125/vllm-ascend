package cn.nextapp.platform;

import java.net.URLDecoder;
import java.net.URLEncoder;

import javax.servlet.http.Cookie;

import my.mvc.RequestContext;
import my.util.CryptUtils;

import org.apache.commons.codec.binary.Base64;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang.math.NumberUtils;

/**
 * 用户登录信息管理器
 * @author Winter Lau
 * @date 2011-12-28 下午2:11:07
 */
public class UserLoginManager {

	private final static String UTF_8 = "UTF-8";
	private final static long COOKIE_TIMEOUT = 1800000;//30分钟不动，cookie自动失效，需要重新登录
	
	/**
	 * 获取当前登录的帐号
	 * @param ctx
	 * @return
	 */
	public static int get(RequestContext ctx) {
		int user = -1;
		Cookie cookie = ctx.cookie(COOKIE_LOGIN);
		if(cookie!=null && StringUtils.isNotBlank(cookie.getValue())){
			String ck = decrypt(cookie.getValue());
			final String[] items = StringUtils.split(ck, '|');
			if(items!=null && items.length == 3){
				user = NumberUtils.toInt(items[0],0);
				long lastTime = NumberUtils.toLong(items[2]);
				if((System.currentTimeMillis() - lastTime) > COOKIE_TIMEOUT){
					delete(ctx);
					user = -1;
				}
				else {
					//更新cookie信息
					delete(ctx);
					save(ctx, user);
				}
			}
		}
		return user;
	}

	public static void save(RequestContext ctx, int user) {
		String new_value = _GenLoginKey(user, ctx.ip());
		ctx.deleteCookie(COOKIE_LOGIN, true);
		ctx.cookie(COOKIE_LOGIN,new_value, -1 ,true);//只在本浏览器会话中有效
	}

	/**
	 * 生成用户登录标识字符串
	 * @param user
	 * @param ip
	 * @param user_agent
	 * @return
	 */
	private static String _GenLoginKey(int user, String ip) {
		StringBuilder sb = new StringBuilder();
		sb.append(user);
		sb.append('|');
		sb.append(ip);
		sb.append('|');
		sb.append(System.currentTimeMillis());
		return encrypt(sb.toString());	
	}

	public static void delete(RequestContext ctx) {
		ctx.deleteCookie(COOKIE_LOGIN, true);
	}
	
	/**
	 * 加密
	 * @param value
	 * @return 
	 * @throws Exception 
	 */
	private static String encrypt(String value) {
		return encrypt(value, E_KEY);
	}

	/**
	 * 加密
	 * @param value
	 * @return 
	 * @throws Exception 
	 */
	private static String encrypt(String value, byte[] key) {
		byte[] data = CryptUtils.encrypt(value.getBytes(), key);
		try{
			return URLEncoder.encode(new String(Base64.encodeBase64(data)), UTF_8);
		}catch(Exception e){
			return null;
		}
	}

	/**
	 * 解密
	 * @param value
	 * @return
	 * @throws Exception 
	 */
	private static String decrypt(String value) {
		return decrypt(value, E_KEY);
	}	

	/**
	 * 解密
	 * @param value
	 * @return
	 * @throws Exception 
	 */
	private static String decrypt(String value, byte[] key) {
		try {
			value = URLDecoder.decode(value,UTF_8);
			if(StringUtils.isBlank(value)) return null;
			byte[] data = Base64.decodeBase64(value.getBytes());
			return new String(CryptUtils.decrypt(data, key));
		} catch (Exception excp) {
			return null;
		}
	}	

	private final static String COOKIE_LOGIN = "nextapp_userid";
	private final static byte[] E_KEY = new byte[]{'.','N','e','x','t','A','p','p'};
}
