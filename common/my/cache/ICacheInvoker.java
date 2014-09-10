/**
 * 
 */
package my.cache;

/**
 * 回调接口
 * @author Winter Lau
 */
public interface ICacheInvoker {

	public Object callback(Object old_data, Object...args) ; 
	
}
